# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus
from sklearn.metrics import homogeneity_score, silhouette_score

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_opt.sklearn.data_loader import load_data_for_range, load_data
from nvflare.app_common.app_constant import AppConstants

from dfa_lib_python.dataflow import Dataflow
from dfa_lib_python.transformation import Transformation
from dfa_lib_python.attribute import Attribute
from dfa_lib_python.attribute_type import AttributeType
from dfa_lib_python.set import Set
from dfa_lib_python.set_type import SetType
from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
from dfa_lib_python.task_status import TaskStatus
from dfa_lib_python.extractor_extension import ExtractorExtension
from dfa_lib_python.dependency import Dependency

from time import perf_counter
import datetime

from pathlib import Path


dataflow_tag = "nvidiaflare-df"


class KMeansLearner(Learner):
    def __init__(
        self,
        data_path: str,
        client_id: int,
        hash_trial: str,
        valid_frac: float,
        random_state: int = None,
        max_iter: int = 1,
        n_init: int = 1,
        reassignment_ratio: int = 0,
    ):
        super().__init__()
        self.data_path = data_path
        self.valid_frac = valid_frac
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.reassignment_ratio = reassignment_ratio
        self.client_id = int(client_id)
        self.train_data = None
        self.valid_data = None
        self.n_samples = None
        self.n_clusters = None
        self.hash_trial = hash_trial

    def load_data(self) -> dict:
        t3 = Task(3, dataflow_tag, "LoadData")
        t3.begin()
        start = perf_counter()

        train_data = load_data(self.data_path, require_header=True)
        data_size = train_data[-1]

        valid_size = int(round(data_size * self.valid_frac))

        indices = {
            "valid": {"start": 0, "end": valid_size},
        }
        
        train_data = load_data(
            self.data_path,
        )

        valid_data = load_data_for_range(
            self.data_path,  
            indices["valid"]["start"],
            indices["valid"]["end"],
        )   

        duration = perf_counter() - start
        timestamp = datetime.datetime.now()
        to_dfanalyzer = [self.hash_trial, self.client_id, duration, timestamp]
        t3_input = DataSet("iLoadData", [Element(to_dfanalyzer)])
        t3.add_dataset(t3_input)
        t3_output = DataSet("oLoadData", [Element([])])
        t3.add_dataset(t3_output)
        t3.end()

        return {"train": train_data, "valid": valid_data}

    def initialize(self, parts: dict, fl_ctx: FLContext):
        data = self.load_data()
        self.train_data = data["train"]
        self.valid_data = data["valid"]
        # train data size, to be used for setting
        # NUM_STEPS_CURRENT_ROUND for potential use in aggregation
        self.n_samples = data["train"][-1]
        t4 = Task(4, dataflow_tag, "InitializeClient",
                  dependency=Task(3, dataflow_tag, "LoadData"))
        t4.begin()
        start = perf_counter()
        duration = perf_counter() - start

        timestamp = datetime.datetime.now()
        to_dfanalyzer = [self.hash_trial, self.client_id, self.n_samples, duration, timestamp]
        t4_input = DataSet("iInitializeClient", [Element(to_dfanalyzer)])
        t4.add_dataset(t4_input)
        t4_output = DataSet("oInitializeClient", [Element([])])
        t4.add_dataset(t4_output)
        t4.end()
        # note that the model needs to be created every round
        # due to the available API for center initialization

    def train(
        self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext
    ) -> Tuple[dict, dict]:
        t5 = Task(5 + 4 * (curr_round), dataflow_tag, "ClientTraining")
        if curr_round == 0:
            t5.add_dependency(
                Dependency(
                    ["InitializeClient", "ClientValidation"],
                    ["4", "0"],
                )
            )
        else:
            t5.add_dependency(
                Dependency(
                    ["InitializeClient", "ClientValidation"],
                    ["4", str(8 + 4 * (curr_round - 2))],
                )
            )

        t5.begin()
        start = perf_counter()
        timestamp = datetime.datetime.now()
        to_dfanalyzer = [
            self.hash_trial,
            self.client_id,
            curr_round,
            self.n_clusters,
            self.n_samples,
            self.max_iter,
            self.n_init,
            self.reassignment_ratio,
            self.random_state,
            timestamp
        ]

        t5_input = DataSet("iClientTraining", [Element(to_dfanalyzer)])
        t5.add_dataset(t5_input)

        # get training data, note that clustering is unsupervised
        # so only x_train will be used
        count_local = None
        (x_train, y_train, train_size) = self.train_data

        center_global = None
        if curr_round == 0:
            # first round, compute initial center with kmeans++ method
            # model will be None for this round
            self.n_clusters = global_param["n_clusters"]
            center_local, _ = kmeans_plusplus(
                x_train, n_clusters=self.n_clusters, random_state=self.random_state
            )
            kmeans = None
            params = {"center": center_local, "count": None}
        else:
            center_global = global_param["center"]
            # following rounds, local training starting from global center
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=self.n_samples,
                max_iter=self.max_iter,
                init=center_global,
                n_init=self.n_init,
                reassignment_ratio=self.reassignment_ratio,
                random_state=self.random_state,
            )
            kmeans.fit(x_train)
            center_local = kmeans.cluster_centers_
            count_local = kmeans._counts
            params = {"center": center_local, "count": count_local}

        duration = perf_counter() - start
        timestamp = datetime.datetime.now()

        to_dfanalyzer = [
            self.hash_trial,
            self.client_id,
            curr_round,
            center_local,
            count_local,
            center_global,
            duration,
            timestamp
        ]

        t5_output = DataSet("oClientTraining", [Element(to_dfanalyzer)])
        t5.add_dataset(t5_output)
        t5.end()

        return params, kmeans

    def validate(
        self, curr_round: int, global_param: Optional[dict], fl_ctx: FLContext
    ) -> Tuple[dict, dict]:
        # local validation with global center
        # fit a standalone KMeans with just the given center

        t7 = Task(
            8 + 4 * (curr_round-1),
            dataflow_tag,
            "ClientValidation",
            dependency=Task(7 + 4 * (curr_round-1), dataflow_tag, "Assemble"),
        )
        t7.begin()
        start = perf_counter()
        timestamp = datetime.datetime.now()
        to_dfanalyzer = [self.hash_trial, self.client_id, curr_round, timestamp]
        t7_input = DataSet("iClientValidation", [Element(to_dfanalyzer)])
        t7.add_dataset(t7_input)

        center_global = global_param["center"]
        kmeans_global = KMeans(n_clusters=self.n_clusters, init=center_global, n_init=1)
        kmeans_global.fit(center_global)
        # get validation data, both x and y will be used
        (x_valid, y_valid, valid_size) = self.valid_data
        y_pred = kmeans_global.predict(x_valid)
        silhouette = silhouette_score(x_valid, y_pred)
        self.log_info(fl_ctx, f"Silhouette Score {silhouette:.4f}")
        metrics = {"Silhouette Score": silhouette}

        duration = perf_counter() - start
        timestamp = datetime.datetime.now()

        to_dfanalyzer = [self.hash_trial, self.client_id, curr_round, silhouette, duration, timestamp]
        
        t7_output = DataSet("oClientValidation", [Element(to_dfanalyzer)])
        t7.add_dataset(t7_output)
        t7.end()
        return metrics, kmeans_global

    def finalize(self, fl_ctx: FLContext) -> None:
        # freeing resources in finalize
        # get round
        curr_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        timestamp = datetime.datetime.now()

        t9 = Task(
            9 + 4 * (curr_round),
            dataflow_tag,
            "FinalizeClient",
            dependency=Task(8 + 4 * (curr_round-1), dataflow_tag, "ClientValidation"),
        )
        t9.begin()
        start = perf_counter()
        del self.train_data
        del self.valid_data
        self.log_info(fl_ctx, "Freed training resources")

        duration = perf_counter() - start
        to_dfanalyzer = [self.hash_trial, self.client_id, duration, timestamp]
        t9_output = DataSet("oFinalizeClient", [Element(to_dfanalyzer)])
        t9.add_dataset(t9_output)
        t9.end()
