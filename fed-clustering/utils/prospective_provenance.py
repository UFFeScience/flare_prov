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

# DfAnalyzer Instrumentation
dataflow_tag = "nvidiaflare-df"

df = Dataflow(dataflow_tag)

tf1 = Transformation("PrepareData")
tf1_input = Set(
    "iPrepareData",
    SetType.INPUT,
    [
        Attribute("trial_id", AttributeType.TEXT),
        Attribute("dataset_name", AttributeType.TEXT),
        Attribute("randomize", AttributeType.NUMERIC),
        Attribute("out_path", AttributeType.TEXT),
        Attribute("duration", AttributeType.NUMERIC),
    ],
)
tf1_output = Set("oPrepareData", SetType.OUTPUT, [])
tf1.set_sets([tf1_input, tf1_output])
df.add_transformation(tf1)

tf2 = Transformation("JobConfig")

tf2_input = Set(
    "iJobConfig",
    SetType.INPUT,
    [
        Attribute("trial_id", AttributeType.TEXT),
        Attribute("task_name", AttributeType.TEXT),
        Attribute("data_path", AttributeType.TEXT),
        Attribute("site_num", AttributeType.NUMERIC),
        Attribute("site_name_prefix", AttributeType.TEXT),
        Attribute("data_size", AttributeType.NUMERIC),
        Attribute("valid_frac", AttributeType.NUMERIC),
        Attribute("split_method", AttributeType.TEXT),
        Attribute("num_rounds", AttributeType.NUMERIC),
    ],
)
tf2_output = Set("oJobConfig", SetType.OUTPUT, [])

tf1_output.set_type(SetType.INPUT)
tf1_output.dependency = tf1._tag

tf2.set_sets([tf1_output, tf2_input, tf2_output])
df.add_transformation(tf2)

tf3 = Transformation("LoadData")
tf3_input = Set(
    "iLoadData",
    SetType.INPUT,
    [
        Attribute("trial_id", AttributeType.TEXT),
        Attribute("client_id", AttributeType.TEXT),
        Attribute("loading_time", AttributeType.NUMERIC),
        Attribute("timestamp", AttributeType.TEXT),
    ],
)

tf3_output = Set("oLoadData", SetType.OUTPUT, [])

tf2_output.set_type(SetType.INPUT)
tf2_output.dependency = tf2._tag

tf3.set_sets([tf2_output, tf3_input, tf3_output])
df.add_transformation(tf3)

tf4 = Transformation("InitializeClient")
tf4_input = Set(
    "iInitializeClient",
    SetType.INPUT,
    [
        Attribute("trial_id", AttributeType.TEXT),
        Attribute("client_id", AttributeType.TEXT),
        Attribute("n_samples", AttributeType.NUMERIC),
        Attribute("duration", AttributeType.NUMERIC),
        Attribute("timestamp", AttributeType.TEXT),
    ],
)

tf4_output = Set(
    "oInitializeClient",
    SetType.OUTPUT,
    [],
)

tf3_output.set_type(SetType.INPUT)
tf3_output.dependency = tf3._tag

tf4.set_sets([tf3_output, tf4_input, tf4_output])

df.add_transformation(tf4)

tf5 = Transformation("ClientTraining")
tf5_input = Set(
    "iClientTraining",
    SetType.INPUT,
    [
        Attribute("trial_id", AttributeType.TEXT),
        Attribute("client_id", AttributeType.TEXT),
        Attribute("current_round", AttributeType.NUMERIC),
        Attribute("n_clusters", AttributeType.TEXT),
        Attribute("batch_size", AttributeType.TEXT),
        Attribute("max_iter", AttributeType.NUMERIC),
        Attribute("n_init", AttributeType.NUMERIC),
        Attribute("reassignment_ratio", AttributeType.NUMERIC),
        Attribute("random_state", AttributeType.TEXT),
        Attribute("timestamp", AttributeType.TEXT),
    ],
)
tf5_output = Set(
    "oClientTraining",
    SetType.OUTPUT,
    [
        Attribute("trial_id", AttributeType.TEXT),
        Attribute("client_id", AttributeType.TEXT),
        Attribute("current_round", AttributeType.NUMERIC),
        Attribute("center_local", AttributeType.TEXT),
        Attribute("count_local", AttributeType.TEXT),
        Attribute("center_global", AttributeType.TEXT),
        Attribute("training_time", AttributeType.NUMERIC),
        Attribute("timestamp", AttributeType.TEXT),

    ],
)


tf4_output.set_type(SetType.INPUT)
tf4_output.dependency = tf4._tag


tf5.set_sets([tf4_output, tf5_input, tf5_output])

df.add_transformation(tf5)

tf6 = Transformation("GetModelParams")
tf6_input = Set(
    "iGetModelParams",
    SetType.INPUT,
    [
        Attribute("trial_id", AttributeType.TEXT),
        Attribute("center", AttributeType.TEXT),
        Attribute("count", AttributeType.TEXT),
    ],
)

tf6_output = Set(
    "oGetModelParams",
    SetType.OUTPUT,
    [],
)

tf5_output.set_type(SetType.INPUT)
tf5_output.dependency = tf5._tag


tf6.set_sets([tf5_output, tf6_input, tf6_output])

df.add_transformation(tf6)

tf7 = Transformation("Assemble")

tf7_input = Set(
    "iAssemble",
    SetType.INPUT,
    [
        Attribute("trial_id", AttributeType.TEXT),
        Attribute("current_round", AttributeType.NUMERIC),
        Attribute("n_feature", AttributeType.NUMERIC),
        Attribute("n_cluster", AttributeType.NUMERIC),
        Attribute("timestamp", AttributeType.TEXT),

    ],
)
tf7_output = Set(
    "oAssemble",
    SetType.OUTPUT,
    [
        Attribute("trial_id", AttributeType.TEXT),
        Attribute("center", AttributeType.TEXT),
        Attribute("count", AttributeType.TEXT),
        Attribute("assembling_time", AttributeType.NUMERIC),
        Attribute("minibatch_kmeans_time", AttributeType.NUMERIC),
        Attribute("timestamp", AttributeType.TEXT),  
    ],
)

tf6_output.set_type(SetType.INPUT)
tf6_output.dependency = tf6._tag

tf7.set_sets(
    [
        tf6_output,
        tf7_input,
        tf7_output,
    ]
)
df.add_transformation(tf7)

tf8 = Transformation("ClientValidation")
tf8_input = Set(
    "iClientValidation",
    SetType.INPUT,
    [
        Attribute("trial_id", AttributeType.TEXT),
        Attribute("client_id", AttributeType.TEXT),
        Attribute("current_round", AttributeType.NUMERIC),
        Attribute("timestamp", AttributeType.TEXT),

    ],
)
tf8_output = Set(
    "oClientValidation",
    SetType.OUTPUT,
    [
        Attribute("trial_id", AttributeType.TEXT),
        Attribute("client_id", AttributeType.TEXT),
        Attribute("current_round", AttributeType.NUMERIC),
        Attribute("silhouette_score", AttributeType.NUMERIC),
        Attribute("validation_time", AttributeType.NUMERIC),
        Attribute("timestamp", AttributeType.TEXT),

    ],
)

tf7_output.set_type(SetType.INPUT)
tf7_output.dependency = tf7._tag

tf8.set_sets([tf7_output, tf8_input, tf8_output])
df.add_transformation(tf8)


tf5 = Transformation("ClientTraining")

tf8_output.set_type(SetType.INPUT)
tf8_output.dependency = tf8._tag

tf5.set_sets([tf8_output])
df.add_transformation(tf5)

tf9 = Transformation("FinalizeClient")
tf9_output = Set(
    "oFinalizeClient",
    SetType.OUTPUT,
    [
        Attribute("trial_id", AttributeType.TEXT),
        Attribute("client_id", AttributeType.TEXT),
        Attribute("duration", AttributeType.NUMERIC),
        Attribute("timestamp", AttributeType.TEXT),

    ],
)

# tf8_output.set_type(SetType.INPUT)
# tf8_output.dependency = tf8._tag

tf9.set_sets([tf8_output, tf9_output])
df.add_transformation(tf9)

df.save()
