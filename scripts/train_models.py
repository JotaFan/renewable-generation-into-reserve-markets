import json
import os
import pathlib
import sys

import keras
import numpy as np
import pandas as pd
from forecat import CNNArch, DenseArch, LSTMArch, UNETArch,EncoderDecoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

scripts_path = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, scripts_path) # Add the script's directory to the system path

from generator import get_dataset

from utils import CustomModelCheckpoint, StopOnNanLoss, ModelCheckpoint, SaveModelCallback
from predictions_scores import prediction_score, save_scores


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



models_strutres = {
    "VanillaCNN":{"arch":CNNArch,}, 
    "VanillaDense":{"arch":DenseArch}, 
    "VanillaLSTM":{"arch":LSTMArch},
    "StackedCNN":{"arch":CNNArch,  "architeture_args":{"block_repetition":2}}, 
    #"StackedCNN8":{"arch":CNNArch,  "architeture_args":{"block_repetition":8}}, 
    "StackedLSTMA":{"arch":LSTMArch,  "architeture_args":{"block_repetition":2}},
    "UNET" : {"arch":UNETArch},
    "EncoderDecoder":{"arch":EncoderDecoder}
}


def train_save_model(dataset, model_name,struct_name,get_dataset_args={},
epocas=10, loss="mse", model_name_load=None):
    if model_name_load is None:
        model_name_load = model_name
    model_folder = os.path.join(path_to_trained_models_folder, struct_name, model_name)
    os.makedirs(model_folder, exist_ok=True)
    dataset_to_use = dataset.copy()


    train_dataset_X, train_dataset_Y, test_dataset_X, test_dataset_Y, gen = get_dataset(dataset_to_use,**get_dataset_args)

    # All multihead is inside, so no list, and strict policy of:
    # X = (N, time, features)
    n_features_train = train_dataset_X.shape[-1]
    n_features_predict = 1

    input_args = {
        "X_timeseries": get_dataset_args["time_moving_window_size_X"],
        "Y_timeseries": get_dataset_args["time_moving_window_size_Y"],
        "n_features_train": n_features_train,
        "n_features_predict": n_features_predict,
    }

    model_conf =models_strutres[model_name_load]
    architeture_args = model_conf.get("architeture_args", {})
    keras.backend.clear_session()

    forearch = model_conf["arch"](**input_args)
    foremodel = forearch.architeture(**architeture_args)
    metrics = keras.metrics.RootMeanSquaredError()
    
    model_keras_filename = os.path.join(model_folder, f"{model_name}.keras")
    #if os.path.exists(model_keras_filename):
     #   return
    history=None
    period = 10
    model_checkpoint = keras.callbacks.ModelCheckpoint(model_keras_filename)
    freq_saves_folder = model_keras_filename.replace(".keras","freq_saves") 
    os.makedirs(freq_saves_folder, exist_ok=True)
    already_trained = os.listdir(freq_saves_folder)
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

    save_on_freq = SaveModelCallback(period, frq_model_filename_sof, model_history_filename, logs=history,start_epoch=max_trained_epocas)
    
    STEPS_PER_EPOCH = 2336

    custom_model_checkpoint = CustomModelCheckpoint(frq_model_filename, period, max_trained_epocas)
    callbacks = [model_checkpoint,save_on_freq, StopOnNanLoss(model_keras_filename),]

    foremodel.compile(optimizer="adam", 
        loss=loss,
        metrics=metrics,) 

    history_new = foremodel.fit(
                    train_dataset_X,
                    train_dataset_Y,
                    epochs=epocas,
                    callbacks=callbacks
                )

    history_to_save = {}
    if history:
        if isinstance(history, dict):
            for key in history:
                old = history[key]
                new = history_new.history.get(key, [])
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

    predict_score = prediction_score(test_dataset_Y, predictions, test_allocation, model_name=model_name)

    save_scores(test_dataset_Y, predictions, test_allocation, model_test_filename, predict_score, model_score_filename)



    return


# Experiment 1 - Epocas and Archs
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
# for model_name in models_strutres:
#     print(model_name)
#     try:
#         train_save_model(dataset, model_name,struct_name,get_dataset_args=get_dataset_args, epocas=epocas)
#     except Exception as e:
#         print(e)

# Experiment 2 - losses
epocas=30

X_timeseries = 168
Y_timeseries = 24
frac = 0.95
train_features_folga = 24
skiping_step=1
keep_y_on_x=True

struct_name = "losses_experiment"
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
    "StackedCNN":{"arch":CNNArch,  "architeture_args":{"block_repetition":2}}, 
    "UNET" : {"arch":UNETArch},
}

from losses import MeanSquaredLogarithmicError, weighted_loss, mean_squared_diff_error, mean_absolute_percentage_error


losses = {#"mae":"mae", "mse":"mse", 
"msle":MeanSquaredLogarithmicError(), 
"wl":weighted_loss, #"msde":mean_squared_diff_error, "mape":mean_absolute_percentage_error
}

for loss_name,loss in losses.items():
    for model_name in models_strutres:
        model_name_to_save = model_name + loss_name
        print(model_name_to_save)
        try:
            train_save_model(dataset, model_name_to_save,struct_name,get_dataset_args=get_dataset_args, epocas=epocas, loss=loss, model_name_load=model_name)
        except Exception as e:
            print(e)
            
# Experiment 3 - activation end

# Experiment 4 - time windows

# Experiment 5 - clusterings