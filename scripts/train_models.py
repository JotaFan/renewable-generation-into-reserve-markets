import json
import os
import pathlib
import sys
import math
import keras_core as keras
from keras_core.metrics import RootMeanSquaredError
import numpy as np
import pandas as pd
from forecat import CNNArch, DenseArch, LSTMArch, UNETArch,EncoderDecoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import tensorflow as tf
import traceback

scripts_path = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, scripts_path) # Add the script's directory to the system path

from generator import get_dataset

from utils import StopOnNanLoss, ModelCheckpoint, SaveModelCallback
from predictions_scores import prediction_score, save_scores

def assign_labels_with_limits(values, classes_dict):
    labels = []
    last_label = sorted([f for f in classes_dict.keys()])[-1]
    for value in values:
        for label, limits in classes_dict.items():
            if len(limits) == 1:
                if value == 0:
                    labels.append(0)
                    break
                elif label == 1:
                    if value <= limits[0]:
                        labels.append(label)
                elif label == last_label:
                    if value >= limits[0]:
                        labels.append(label)                    
            elif len(limits) == 2 and limits[0] <= value <= limits[1]:
                labels.append(label)

    return np.array(labels)



def get_freq_samples(train_dataset_labels ):
    unique_values, value_counts = np.unique(train_dataset_labels, return_counts=True, axis=None)
    values_dict = {}
    for u, c in zip(unique_values, value_counts):
        values_dict[u]=c


    # Define a function to map values using the dictionary
    map_func = np.vectorize(lambda x: values_dict.get(x, x))

    # Apply the function to arrayA to create arrayB
    sample_weigths = map_func(train_dataset_labels)
    sample_weigths = 1/sample_weigths
    return sample_weigths

dataset_file_path = "../data/dados_2014-2022.csv"
dataset_file_path = pathlib.Path(scripts_path, dataset_file_path).resolve()

dataset = pd.read_csv(dataset_file_path, index_col=0)

columns_Y = ["UpwardUsedSecondaryReserveEnergy"]
alloc_column = ["SecondaryReserveAllocationAUpward"]
y_columns = columns_Y
datetime_col = "datetime"


d = pd.to_datetime(dataset[datetime_col], format="mixed", utc=True)
columns_X = dataset.columns[~dataset.columns.isin(columns_Y)]

mean = [t for f,t in dataset[columns_Y].copy().replace(0, np.nan).mean().items()]






dataset["hour"] = [f.hour for f in d]
dataset["day"] = [f.day for f in d]
dataset["month"] = [f.month for f in d]
dataset["year"] = [f.year for f in d]
dataset["day_of_year"] = [f.timetuple().tm_yday for f in d]
dataset["day_of_week"] = [f.timetuple().tm_wday for f in d]
dataset["week_of_year"] = [f.weekofyear for f in d]



time_cols = ["hour", "day", "month", "year", "day_of_year", "day_of_week", "week_of_year"]
# Make the y the 1st column
dataset = dataset[y_columns+[col for col in dataset.columns if col not in y_columns]]

# make the time columns the last
dataset = dataset[[col for col in dataset.columns if col not in time_cols]+time_cols]


df = dataset.copy()
df.drop("datetime", axis=1, inplace=True)
# Sort DataFrame by DateTime index
df.sort_index(inplace=True)

# Perform imputation
imputer = IterativeImputer(max_iter=10, random_state=0)
df_imputed = imputer.fit_transform(df)

# Convert the result back to a DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)
df_imputed["datetime"] = dataset["datetime"]
dataset = df_imputed.copy()
df_imputed = None


path_to_trained_models_folder = "../models_trained/"
path_to_trained_models_folder = pathlib.Path(scripts_path, path_to_trained_models_folder).resolve()

from alquitable.losses import mean_squared_error

models_strutres = {
    "VanillaCNN":{"arch":CNNArch,}, 
    "VanillaDense":{"arch":DenseArch}, 
    "VanillaLSTM":{"arch":LSTMArch},
    "StackedCNN":{"arch":CNNArch,  "architecture_args":{"block_repetition":2}}, 
    #"StackedCNN8":{"arch":CNNArch,  "architecture_args":{"block_repetition":8}}, 
    "StackedLSTMA":{"arch":LSTMArch,  "architecture_args":{"block_repetition":2}},
    "UNET" : {"arch":UNETArch},
    "EncoderDecoder":{"arch":EncoderDecoder}
}
models_strutres_DEFAULT = models_strutres

def train_save_model(dataset, model_name,struct_name,get_dataset_args={},
epocas=10, loss="mse", model_name_load=None, models_strutres=None, input_args={}, classes_dict={}, optimizer="adam",
weight=None,batch_size=252,
):

    if models_strutres is None:
        models_strutres = models_strutres_DEFAULT
    if model_name_load is None:
        model_name_load = model_name
    model_folder = os.path.join(path_to_trained_models_folder, struct_name, model_name)
    os.makedirs(model_folder, exist_ok=True)
    mean = np.nanmean(dataset[get_dataset_args["y_columns"]].values)
    dataset_to_use = dataset.copy()
    train_dataset_Y_values = None

    train_dataset_X, train_dataset_Y, test_dataset_X, test_dataset_Y, gen = get_dataset(dataset_to_use,**get_dataset_args)

    # All multihead is inside, so no list, and strict policy of:
    # X = (N, time, features)
    n_features_train = train_dataset_X.shape[-1]
    n_features_predict = 1

    input_args.update({
        "X_timeseries": get_dataset_args["time_moving_window_size_X"],
        "Y_timeseries": get_dataset_args["time_moving_window_size_Y"],
        "n_features_train": n_features_train,
        "n_features_predict": n_features_predict,
    })

    model_conf =models_strutres[model_name_load]
    architecture_args = model_conf.get("architecture_args", {})
    keras.backend.clear_session()

    forearch = model_conf["arch"](**input_args)
    foremodel = forearch.architecture(**architecture_args)
    metrics = ["root_mean_squared_error"]#{"root_mean_squared_error" :keras.metrics.RootMeanSquaredError()} root_mean_squared_error
    
    model_keras_filename = os.path.join(model_folder, f"{model_name}.keras")
    #if os.path.exists(model_keras_filename):
    #   return
    history=None
    period = 1
    # if "linear_models_epocs" in struct_name:
    #     if "Enco" in model_name:
    #         period = 1
    #         return
    #     if "LSTM" in model_name:
    #         period = 1
    #         return
    # if "UNET" in model_name:
    #     period = 1
    #     return

    model_checkpoint = keras.callbacks.ModelCheckpoint(model_keras_filename)
    freq_saves_folder = model_keras_filename.replace(".keras","freq_saves") 
    os.makedirs(freq_saves_folder, exist_ok=True)
    already_trained = os.listdir(freq_saves_folder)
    already_trained = [f for f in already_trained if "unfi" not in f]
    max_trained_epocas = [int(os.path.basename(f).replace(".keras", "")) for f in already_trained]
    model_history_filename = os.path.join(model_folder, f"{model_name}_history.json")

    if len(max_trained_epocas)!=0:
        max_trained_epocas = max(max_trained_epocas)
    else:
        max_trained_epocas = 0
    if epocas <= max_trained_epocas:
        return
    elif max_trained_epocas!=0:
        last_trained_model_path = [f for f in already_trained if int(os.path.basename(f).replace(".keras", ""))==max_trained_epocas][0]
        last_trained_model_path = os.path.join(freq_saves_folder, last_trained_model_path)
        foremodel = keras.models.load_model(last_trained_model_path)
        epocas = epocas - max_trained_epocas
        if os.path.exists(model_history_filename):
            with open(model_history_filename) as f:
                history = json.load(f)

    frq_model_filename = model_keras_filename.replace(".keras", "freq_saves/{epoch:02d}.keras")
    frq_model_filename_sof = model_keras_filename.replace(".keras", "freq_saves/{epoch}.keras")

    #save_on_freq = SaveModelCallback(period, frq_model_filename_sof, model_history_filename, logs=history,start_epoch=max_trained_epocas)
    
    STEPS_PER_EPOCH = 2336

    callbacks = [model_checkpoint, StopOnNanLoss(model_keras_filename, model_log_filename=model_history_filename, logs=history,
        save_frequency=period,start_epoch=max_trained_epocas, model_keras_filename=frq_model_filename_sof),]

 

    # Checnk for multioupt
    output_shape = foremodel.outputs
    if isinstance(output_shape, list):
        if len(output_shape)>1:
            clsuter_shape = (len(train_dataset_Y), *tuple(output_shape[1].shape[-2:]))
            clsuter_shape_test = (len(test_dataset_Y), *tuple(output_shape[1].shape[-2:]))

            # make labels
            train_dataset_labels = assign_labels_with_limits(train_dataset_Y.ravel(), classes_dict)
            # pass to one hoe
            train_dataset_labels = np.array(tf.one_hot(train_dataset_labels, np.max(train_dataset_labels)+1)).reshape(clsuter_shape)

            test_dataset_labels = assign_labels_with_limits(test_dataset_Y.ravel(), classes_dict)
            test_dataset_labels = np.array(tf.one_hot(test_dataset_labels, np.max(test_dataset_labels)+1)).reshape(clsuter_shape_test)
            if len(output_shape)==2:
                train_dataset_Y_values = train_dataset_Y
                test_dataset_Y_values = test_dataset_Y

                train_dataset_Y = train_dataset_Y, train_dataset_labels
                test_dataset_Y = test_dataset_Y, test_dataset_labels

                loss = [loss, "categorical_crossentropy"]
                metrics_keys = [str(node.name).split("/")[0] for node in foremodel.outputs]
                metrics = {
                    metrics_keys[0]:"root_mean_squared_error",
                        metrics_keys[1]:"categorical_accuracy",
                }
            elif len(output_shape)==3:
                train_dataset_Y_values = train_dataset_Y
                test_dataset_Y_values = test_dataset_Y

                train_dataset_Y = train_dataset_Y, train_dataset_labels, train_dataset_Y
                test_dataset_Y = test_dataset_Y, test_dataset_labels, test_dataset_Y
                loss = [loss, "categorical_crossentropy", loss]
                metrics_keys = [str(node.name).split("/")[0] for node in foremodel.outputs]
                metrics = {
                    metrics_keys[0]:"root_mean_squared_error",
                        metrics_keys[1]:"categorical_accuracy",
                    metrics_keys[2]:"root_mean_squared_error",
                }




    compile_args = {"optimizer":optimizer, 
        "loss":loss,
        "metrics":metrics,}

    foremodel.compile(**compile_args) 

    fit_args={
        "epochs":epocas,
        "callbacks":callbacks,
        "batch_size":batch_size,
    }

    if weight:
        if "delta_mean" in  weight or "both" in weight:
            train_dataset_Y_values = train_dataset_Y_values or train_dataset_Y
            samples_weights = np.abs(train_dataset_Y_values - mean)
            fit_args["sample_weight"] = samples_weights
        if "freq" in  weight or "both" in weight:
            freq_weights = get_freq_samples(train_dataset_labels)
            fit_args["sample_weight"] = freq_weights
        if "both" in weight:
            fit_args["sample_weight"] = freq_weights*samples_weights
    foremodel.summary()

    history_new = foremodel.fit(
                    train_dataset_X,
                    train_dataset_Y,
                    **fit_args
                )

    history_to_save = {}
    if history:
        if isinstance(history, dict):
            for key in history:
                old = history[key]
                new = history_new.history.get(key, [])
                if not isinstance(old, list):
                    old=[old]
                old = [f for f in old if f]
                old = [f for f in old if not  math.isnan(f)]
                history_to_save[key] = old + new
    for key in history_new.history:
        if key not in history_to_save:
            history_to_save[key] = history_new.history[key]

    
    model_test_filename = os.path.join(model_folder, f"{model_name}_test.npz")
    model_score_filename = os.path.join(model_folder, f"{model_name}_score.json")
    os.path.join(model_folder, f"{model_name}_alloc.npz")

    foremodel.save(model_keras_filename)
    frq_model_filename = frq_model_filename.replace("freq_saves/{epoch:02d}.keras", f"freq_saves/{epocas}.keras")

    if not os.path.exists(frq_model_filename.replace(f"freq_saves/{epocas}.keras", "freq_saves/unfinished.keras")):
        foremodel.save(frq_model_filename)

    with open(model_history_filename, "w") as mfile:
        json.dump(history_to_save, mfile)

    predictions = foremodel.predict(test_dataset_X)

    if np.any(np.isnan(predictions)):
        return

    get_dataset_args["y_columns"] = alloc_column
    test_allocation = get_dataset(dataset_to_use,**get_dataset_args)
    test_allocation = test_allocation[3]
    if isinstance(output_shape, list):
        if len(output_shape)==2:
            test_allocation = test_allocation, assign_labels_with_limits(test_allocation, classes_dict)
        elif len(output_shape)==3:
            test_allocation = test_allocation, assign_labels_with_limits(test_allocation, classes_dict), test_allocation



    predict_score = prediction_score(test_dataset_Y, predictions, test_allocation, model_name=model_name)

    save_scores(test_dataset_Y, predictions, test_allocation, model_test_filename, predict_score, model_score_filename)



    return


# # Experiment 1 - Epocas and Archs
# epocas=200
# X_timeseries = 168
# Y_timeseries = 24
# frac = 0.95
# train_features_folga = 24
# skiping_step=1
# keep_y_on_x=True
# struct_name = "linear_models_epocs"
# get_dataset_args={
#     "y_columns":columns_Y,
#     "time_moving_window_size_X":X_timeseries,
#     "time_moving_window_size_Y":Y_timeseries,
#     "frac":frac,
#     "keep_y_on_x":keep_y_on_x,
#     "train_features_folga":train_features_folga,        
#     "skiping_step":skiping_step,
#     "time_cols":time_cols,
#     "alloc_column":alloc_column,
# }
# input_args = {"activation_middle":"relu",
#             "activation_end":"relu"}
# for model_name in models_strutres:
#     print(model_name)
#     try:
#         train_save_model(dataset, model_name,struct_name,get_dataset_args=get_dataset_args, epocas=epocas, input_args=input_args)
#     except Exception as e:
#         print(f"Exception: {e}")
#         traceback.print_exc()
print("doing the lossssssssssssssssssssssssssssseeeeeeeeeeeeees")
# Experiment 2 - losses
# epocas=150

# X_timeseries = 168
# Y_timeseries = 24
# frac = 0.95
# train_features_folga = 24
# skiping_step=1
# keep_y_on_x=True

# struct_name = "losses_experiment"
# get_dataset_args={

#     "y_columns":columns_Y,
#     "time_moving_window_size_X":X_timeseries,
#     "time_moving_window_size_Y":Y_timeseries,
#     "frac":frac,
#     "keep_y_on_x":keep_y_on_x,
#     "train_features_folga":train_features_folga,        
#     "skiping_step":skiping_step,
#     "time_cols":time_cols,
#     "alloc_column":alloc_column,


# }

# models_strutres = {
#     "UNET" : {"arch":UNETArch},
#     "StackedCNN":{"arch":CNNArch,  "architecture_args":{"block_repetition":2}},
# #    "VanillaCNN":{"arch":CNNArch,}, 
# }

from alquitable.losses import weighted_loss, mean_absolute_percentage_error, mean_squared_diff_error, mean_squared_error
from keras_core.losses import MeanSquaredLogarithmicError

# losses = {"mae":"mae", "mse":"mse", 
# "msle":MeanSquaredLogarithmicError(), 
# "wl":weighted_loss, "msde":mean_squared_diff_error, "mape":mean_absolute_percentage_error
# }

# for loss_name,loss in losses.items():
#     for model_name in models_strutres:
#         model_name_to_save = model_name + loss_name
#         print(model_name_to_save)
#         try:
#             train_save_model(dataset, model_name_to_save,struct_name,get_dataset_args=get_dataset_args, epocas=epocas, loss=loss, model_name_load=model_name)
#         except Exception as e:
#             print(f"Exception: {e}")
#             traceback.print_exc()
            
print("doin the activation!!!!")
# Experiment 3 - activation middle and end
epocas=150
X_timeseries = 168
Y_timeseries = 24
frac = 0.95
train_features_folga = 24
skiping_step=1
keep_y_on_x=True
struct_name = "linear_models_activation"
get_dataset_args={
    "y_columns":columns_Y,
    "time_moving_window_size_X":X_timeseries,
    "time_moving_window_size_Y":Y_timeseries,
    "frac":frac,
    "keep_y_on_x":keep_y_on_x,
    "train_features_folga":train_features_folga,        
    "skiping_step":skiping_step,
    "time_cols":time_cols,
    "alloc_column":alloc_column,
}
models_strutres = {
    "StackedCNN":{"arch":CNNArch,  "architecture_args":{"block_repetition":2}}, 
}

regression_activations = ["relu", "linear", "softplus", "softsign", "tanh", "selu", "elu", "exponential"
]
losses = {#"mae":"mae", "mse":"mse", 
#"msle":MeanSquaredLogarithmicError(), 
"wl":weighted_loss, #"msde":mean_squared_diff_error, "mape":mean_absolute_percentage_error
}
for loss_name,loss in losses.items():

    for middle in regression_activations:
        for end in regression_activations:
            for model_name in models_strutres:
                model_name_to_save = model_name +loss_name+ f"_{middle}_{end}"
                print(model_name_to_save)

                input_args = {"activation_middle":middle,
                "activation_end":end}
                try:
                    train_save_model(dataset, model_name_to_save,
                    struct_name,get_dataset_args=get_dataset_args, epocas=epocas,loss=loss,
                    model_name_load=model_name, input_args=input_args)
                except Exception as e:
                    print(f"Exception: {e}")
                    traceback.print_exc()

epocas=30
X_timeseries = 168
Y_timeseries = 24
frac = 0.95
train_features_folga = 24
skiping_step=1
keep_y_on_x=True
struct_name = "linear_optimizers"
get_dataset_args={
    "y_columns":columns_Y,
    "time_moving_window_size_X":X_timeseries,
    "time_moving_window_size_Y":Y_timeseries,
    "frac":frac,
    "keep_y_on_x":keep_y_on_x,
    "train_features_folga":train_features_folga,        
    "skiping_step":skiping_step,
    "time_cols":time_cols,
    "alloc_column":alloc_column,
}
models_strutres = {
    "StackedCNN":{"arch":CNNArch,  "architecture_args":{"block_repetition":2}}, 
}

regression_optimizer = [
    "SGD",
    "RMSprop",
    "Adam",
    keras.optimizers.AdamW(),
    "Adadelta",
    "Adagrad",
    "Adamax",
    "Adafactor",
    "Nadam",
    "Ftrl",
]

for opt in regression_optimizer:
    for model_name in models_strutres:
        name_opt = opt
        if not isinstance(opt, str):
            name_opt = opt.name

        model_name_to_save = model_name + f"_{name_opt}"
        print(model_name_to_save)

        #input_args = {"activation_middle":middle,
        #"activation_end":end}
        try:
            train_save_model(dataset, model_name_to_save,
            struct_name,get_dataset_args=get_dataset_args, epocas=epocas, loss=MeanSquaredLogarithmicError(),
            model_name_load=model_name, optimizer=opt)
        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()


sys.exit()

# Experiment 4 - time windows
epocas=30
X_timeseries = 168
Y_timeseries = 24
frac = 0.95
train_features_folga = 24
skiping_step=1
keep_y_on_x=True
struct_name = "linear_models_time_windows"
get_dataset_args={
    "y_columns":columns_Y,
    "time_moving_window_size_X":X_timeseries,
    "time_moving_window_size_Y":Y_timeseries,
    "frac":frac,
    "keep_y_on_x":keep_y_on_x,
    "train_features_folga":train_features_folga,        
    "skiping_step":skiping_step,
    "time_cols":time_cols,
    "alloc_column":alloc_column,
}
models_strutres = {
    "StackedCNN":{"arch":CNNArch,  "architecture_args":{"block_repetition":2}}, 
}
for X in [24, 48, 98, 168]:
    for Y in [1, 4, 8, 12, 24]:
        for model_name in models_strutres:
            model_name_to_save = model_name + f"_{X}X_{Y}Y"
            print(model_name_to_save)

            input_args = {"activation_middle":"linear",
            "activation_end":"softplus"}

            get_dataset_args.update({
                "time_moving_window_size_X":X,
                "time_moving_window_size_Y":Y,                
            })
            try:
                train_save_model(dataset, model_name_to_save,struct_name,get_dataset_args=get_dataset_args, 
                input_args=input_args,
                epocas=epocas, loss=MeanSquaredLogarithmicError(),
                model_name_load=model_name)
            except Exception as e:
                print(f"Exception: {e}")
                traceback.print_exc()



# Experiment 5 - clusterings
epocas=30
X_timeseries = 168
Y_timeseries = 24
frac = 0.95
train_features_folga = 24
skiping_step=1
keep_y_on_x=True
struct_name = "linear_models_clustering"
get_dataset_args={
    "y_columns":columns_Y,
    "time_moving_window_size_X":X_timeseries,
    "time_moving_window_size_Y":Y_timeseries,
    "frac":frac,
    "keep_y_on_x":keep_y_on_x,
    "train_features_folga":train_features_folga,        
    "skiping_step":skiping_step,
    "time_cols":time_cols,
    "alloc_column":alloc_column,
}


list_classes_dict = [

 {
    0 : [0],
    1: [26.2],
    2: [26.3, 194.9],
    3: [195.0],},
{
    0 : [0],
    1: [20.2],
    2: [20.3, 119.2],
    3: [119.3, 331.9],
    4: [332.0, 592.6],
    5: [592.7],}
]

for cla in list_classes_dict:

    models_strutres = {
        "StackedCNNClusters":{"arch":CNNArch,  "architecture_args":{"block_repetition":2,
                                                            "multitail":[{"dense_args":{"activation":"relu"}}, 
                                                            {"dense_args":[{"activation":"relu"}, {
                                                                "filters":Y_timeseries*len(cla),
                                                                "activation":"softmax"}],
                                                            "output_layer_args":{"reshape_shape": (Y_timeseries,len(cla))}
                                                                }],}},
                                                                
        "StackedCNNClusterLinear":{"arch":CNNArch,  "architecture_args":{"block_repetition":2,
                                                                "multitail":[{"dense_args":{"activation":"relu"}}, 
                                                            {"dense_args":[{"activation":"relu"}, {
                                                                "filters":Y_timeseries*len(cla),
                                                                "activation":"softmax"}],
                                                            "output_layer_args":{"reshape_shape": (Y_timeseries,len(cla))}

                                                            },
                                                            {"dense_args":{"activation":"relu"}}],
        }}, 

    }


    for i,model_name in enumerate(models_strutres):
        model_name_to_save = model_name + f"_{len(cla)}"
        print(model_name_to_save)


        try:
            train_save_model(dataset, model_name_to_save,
            struct_name,get_dataset_args=get_dataset_args, epocas=epocas, loss=MeanSquaredLogarithmicError(),models_strutres=models_strutres,
            model_name_load=model_name,  classes_dict=cla)
        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()


# Experiment 6 - Pesos
#  sem peso, dist to mean -> linear
# sem peso, dist to mean, frequ, dmean+freq

# Linear
models_strutres = {
    #"UNET" : {"arch":UNETArch},
    "StackedCNN":{"arch":CNNArch,  "architecture_args":{"block_repetition":2}},
    "VanillaCNN":{"arch":CNNArch,}, 
}
epocas=30
X_timeseries = 168
Y_timeseries = 24
frac = 0.95
train_features_folga = 24
skiping_step=1
keep_y_on_x=True
struct_name = "linear_weights"
get_dataset_args={
    "y_columns":columns_Y,
    "time_moving_window_size_X":X_timeseries,
    "time_moving_window_size_Y":Y_timeseries,
    "frac":frac,
    "keep_y_on_x":keep_y_on_x,
    "train_features_folga":train_features_folga,        
    "skiping_step":skiping_step,
    "time_cols":time_cols,
    "alloc_column":alloc_column,
}

input_args = {"activation_middle":"linear",
            "activation_end":"softplus"}

weitgh_list=["delta_mean", "no_weight"]
for weight in weitgh_list:
    for model_name in models_strutres:
        model_name_to_save = model_name + f"_{weight}"
        print(model_name_to_save)
        try:
            train_save_model(dataset, model_name_to_save,struct_name,get_dataset_args=get_dataset_args, epocas=epocas, input_args=input_args,
            weight=weight, model_name_load=model_name
            )
        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()


# Cluster - Wise
epocas=30
X_timeseries = 168
Y_timeseries = 24
frac = 0.95
train_features_folga = 24
skiping_step=1
keep_y_on_x=True
struct_name = "cluster_weights"
get_dataset_args={
    "y_columns":columns_Y,
    "time_moving_window_size_X":X_timeseries,
    "time_moving_window_size_Y":Y_timeseries,
    "frac":frac,
    "keep_y_on_x":keep_y_on_x,
    "train_features_folga":train_features_folga,        
    "skiping_step":skiping_step,
    "time_cols":time_cols,
    "alloc_column":alloc_column,
}


list_classes_dict = [

 {
    0 : [0],
    1: [26.2],
    2: [26.3, 194.9],
    3: [195.0],},
{
    0 : [0],
    1: [20.2],
    2: [20.3, 119.2],
    3: [119.3, 331.9],
    4: [332.0, 592.6],
    5: [592.7],}
]
weitgh_list=["delta_mean", "no_weight", "freq", "both"]
for weight in weitgh_list:
    for cla in list_classes_dict:

        models_strutres = {
            "StackedCNNClusters":{"arch":CNNArch,  "architecture_args":{"block_repetition":2,
                                                                "multitail":[{"dense_args":{"activation":"relu"}}, 
                                                                {"dense_args":[{"activation":"relu"}, {
                                                                    "filters":Y_timeseries*len(cla),
                                                                    "activation":"softmax"}],
                                                                "output_layer_args":{"reshape_shape": (Y_timeseries,len(cla))}
                                                                    }],}},
                                                                    
            "StackedCNNClusterLinear":{"arch":CNNArch,  "architecture_args":{"block_repetition":2,
                                                                    "multitail":[{"dense_args":{"activation":"relu"}}, 
                                                                {"dense_args":[{"activation":"relu"}, {
                                                                    "filters":Y_timeseries*len(cla),
                                                                    "activation":"softmax"}],
                                                                "output_layer_args":{"reshape_shape": (Y_timeseries,len(cla))}

                                                                },
                                                                {"dense_args":{"activation":"relu"}}],
            }}, 

        }


        for i,model_name in enumerate(models_strutres):
            model_name_to_save = model_name + f"_{len(cla)}_{weight}"
            print(model_name_to_save)


            try:
                train_save_model(dataset, model_name_to_save,
                struct_name,get_dataset_args=get_dataset_args, epocas=epocas, loss=MeanSquaredLogarithmicError(),models_strutres=models_strutres,
                model_name_load=model_name,  classes_dict=cla)
            except Exception as e:
                print(f"Exception: {e}")
                traceback.print_exc()


models_strutres = {
    "VanillaCNN":{"arch":CNNArch,}, 
    "StackedCNN":{"arch":CNNArch,  "architecture_args":{"block_repetition":2}}, 
    #"UNET" : {"arch":UNETArch},
}
# Experiment 6 - filters
epocas=50
X_timeseries = 168
Y_timeseries = 24
frac = 0.95
train_features_folga = 24
skiping_step=1
keep_y_on_x=True
struct_name = "filters_on_conv"
get_dataset_args={
    "y_columns":columns_Y,
    "time_moving_window_size_X":X_timeseries,
    "time_moving_window_size_Y":Y_timeseries,
    "frac":frac,
    "keep_y_on_x":keep_y_on_x,
    "train_features_folga":train_features_folga,        
    "skiping_step":skiping_step,
    "time_cols":time_cols,
    "alloc_column":alloc_column,
}

input_args = {"activation_middle":"linear",
            "activation_end":"softplus"}
for filter_value in [16, 32, 64]:
    for model_name in models_strutres:
        model_name_to_save = model_name + f"_{filter_value}"
        print(model_name_to_save)
        if "architecture_args" not in models_strutres[model_name]:
            models_strutres[model_name]["architecture_args"] = {}
        models_strutres[model_name]["architecture_args"].update({
            "conv_args":{"filters":filter_value}
        })
        try:
            train_save_model(dataset, model_name_to_save,struct_name,get_dataset_args=get_dataset_args, epocas=epocas, input_args=input_args,
            models_strutres=models_strutres, model_name_load=model_name,)
        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()


models_strutres = {
    "StackedCNN":{"arch":CNNArch,  "architecture_args":{"block_repetition":2}}, 
}
# Experiment 7 - batchsize
epocas=50
X_timeseries = 168
Y_timeseries = 24
frac = 0.95
train_features_folga = 24
skiping_step=1
keep_y_on_x=True
struct_name = "models_batch_size"
get_dataset_args={
    "y_columns":columns_Y,
    "time_moving_window_size_X":X_timeseries,
    "time_moving_window_size_Y":Y_timeseries,
    "frac":frac,
    "keep_y_on_x":keep_y_on_x,
    "train_features_folga":train_features_folga,        
    "skiping_step":skiping_step,
    "time_cols":time_cols,
    "alloc_column":alloc_column,
}

input_args = {"activation_middle":"linear",
            "activation_end":"softplus"}
for batch_size in [16, 32, 64, 128, 252, 504, 1008]:
    for model_name in models_strutres:
        model_name_to_save = model_name + f"_{batch_size}"
        print(model_name_to_save)
        try:
            train_save_model(dataset, model_name_to_save,struct_name,get_dataset_args=get_dataset_args, epocas=epocas, input_args=input_args,
            models_strutres=models_strutres, model_name_load=model_name,batch_size=batch_size)
        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()