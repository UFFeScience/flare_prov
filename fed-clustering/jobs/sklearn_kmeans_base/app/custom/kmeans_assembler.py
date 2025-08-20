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

from typing import Dict

import numpy as np
from sklearn.cluster import KMeans

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.aggregators.assembler import Assembler
from nvflare.app_common.app_constant import AppConstants
from dfa_lib_python.dataflow import Dataflow
from dfa_lib_python.transformation import Transformation
from dfa_lib_python.attribute import Attribute
from dfa_lib_python.attribute_type import AttributeType
from dfa_lib_python.set import Set
from dfa_lib_python.set_type import SetType
from dfa_lib_python.task import Task
from dfa_lib_python.dependency import Dependency
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
from dfa_lib_python.task_status import TaskStatus
from dfa_lib_python.extractor_extension import ExtractorExtension
from time import perf_counter
import pickle
import datetime
dataflow_tag = "nvidiaflare-df"


class KMeansAssembler(Assembler):
    def __init__(self, hash_trial: str):
        super().__init__(data_kind=DataKind.WEIGHTS)
        # Aggregator needs to keep record of historical
        # center and count information for mini-batch kmeans
        self.center = None
        self.count = None
        self.n_cluster = 0
        self.current_round = 0
        self.hash_trial = hash_trial

    def get_model_params(self, dxo: DXO):

        t6 = Task(
            6 + 4 * (self.current_round),
            dataflow_tag,
            "GetModelParams",
            dependency=Task(
                5 + 4 * (self.current_round), dataflow_tag, "ClientTraining"
            ),
        )
        t6.begin()
        data = dxo.data

        to_dfanalyzer = [self.hash_trial, data["center"], data["count"]]
        t6_input = DataSet("iGetModelParams", [Element(to_dfanalyzer)])
        t6.add_dataset(t6_input)
        t6_output = DataSet("oGetModelParams", [Element([])])
        t6.add_dataset(t6_output)
        t6.end()

        return {"center": data["center"], "count": data["count"]}

    def assemble(self, data: Dict[str, dict], fl_ctx: FLContext) -> DXO:
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        n_feature = 0
        t8 = Task(
            7 + 4 * (current_round),
            dataflow_tag,
            "Assemble",
            dependency=Task(6 + 4 * (current_round), dataflow_tag, "GetModelParams"),
        )
        t8.begin()
        start = perf_counter()
        kmeans_time = 0
        timestamp_beginning = datetime.datetime.now()

        if current_round == 0:
            # First round, collect the information regarding n_feature and n_cluster
            # Initialize the aggregated center and count to all zero
            client_0 = list(self.collection.keys())[0]
            self.n_cluster = self.collection[client_0]["center"].shape[0]
            n_feature = self.collection[client_0]["center"].shape[1]
            self.center = np.zeros([self.n_cluster, n_feature])
            self.count = np.zeros([self.n_cluster])
            # perform one round of KMeans over the submitted centers
            # to be used as the original center points
            # no count for this round
            center_collect = []
            for _, record in self.collection.items():
                center_collect.append(record["center"])
            centers = np.concatenate(center_collect)
            kmeans_center_initial = KMeans(n_clusters=self.n_cluster)
            kmeans_center_initial.fit(centers)
            self.center = kmeans_center_initial.cluster_centers_
        else:
            # Mini-batch k-Means step to assemble the received centers
            start_kmeans = perf_counter()
            for center_idx in range(self.n_cluster):
                centers_global_rescale = (
                    self.center[center_idx] * self.count[center_idx]
                )
                # Aggregate center, add new center to previous estimate, weighted by counts
                for _, record in self.collection.items():
                    centers_global_rescale += (
                        record["center"][center_idx] * record["count"][center_idx]
                    )
                    self.count[center_idx] += record["count"][center_idx]
                # Rescale to compute mean of all points (old and new combined)
                alpha = 1 / self.count[center_idx]
                centers_global_rescale *= alpha
                # Update the global center
                self.center[center_idx] = centers_global_rescale
            kmeans_time = perf_counter() - start_kmeans



        # Define what you want to save
        model_state = {
            'center': self.center,
            'count': self.count,
            'collection': self.collection,
            'hash_trial': self.hash_trial,
            'n_cluster': self.n_cluster,
        }

        # Save the model to disk
        with open('kmeans_model.pkl', 'wb') as f:
            pickle.dump(model_state, f)

        assembling_time = perf_counter() - start
        timestamp = datetime.datetime.now()
        to_dfanalyzer = [self.hash_trial, current_round, n_feature, self.n_cluster, timestamp_beginning]
        t8_input = DataSet("iAssemble", [Element(to_dfanalyzer)])
        t8.add_dataset(t8_input)
        t8_output = DataSet(
            "oAssemble",
            [Element([self.hash_trial, self.center, self.count, assembling_time, kmeans_time, timestamp])],
        )
        t8.add_dataset(t8_output)
        t8.end()
        params = {"center": self.center}
        dxo = DXO(data_kind=self.expected_data_kind, data=params)

        self.current_round = current_round + 1
        return dxo
