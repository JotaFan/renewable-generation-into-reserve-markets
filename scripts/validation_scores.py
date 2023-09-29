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

from utils import StopOnNanLoss, assign_labels_with_limits
from sys_utils import get_functions_from_experiment
from predictions_scores import prediction_score, save_scores, classificaion_scores


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


dict_classe = { "UpwardUsedSecondaryReserveEnergy_3":{
    0 : [0],
    1: [26.2],
    2: [26.3, 194.9],
    3: [195.0],},
    "UpwardUsedSecondaryReserveEnergy_5":{
    0 : [0],
    1: [20.2],
    2: [20.3, 119.2],
    3: [119.3, 331.9],
    4: [332.0, 592.6],
    5: [592.7],}}



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

path_to_validation_folder = "../models_validation/"
path_to_validation_folder = pathlib.Path(scripts_path, path_to_validation_folder).resolve()

path_to_plots_folder = "../plots/"
path_to_plots_folder = pathlib.Path(scripts_path, path_to_plots_folder).resolve()

# Validation is done on 2021 !
mask_2021 = dataset["year"]==2021
mask_last_week_2020 = dataset["year"]==2020
week_hours_plus_day = (24*7)
mmm = mask_last_week_2020[mask_last_week_2020].iloc[-week_hours_plus_day:]
mmme = mmm & mask_last_week_2020
mask_data = mmme | mask_2021

test_dataset = dataset[mask_2021].copy().iloc[:np.sum(mask_data)]#8712
dataset = dataset[mask_data]





schema_list = [filename for filename in os.listdir(path_to_trained_models_folder) if os.path.isdir(os.path.join(path_to_trained_models_folder,filename))]

# Experiment ONE
schema_to_validate = ["linear_weights","linear_models_clustering",
    "linear_models_epocs", "losses_experiment", "linear_models_activation", 
                        "linear_optimizers",
                        "linear_models_time_windows",
                        ]

def make_validation_epoch_scores(model_experiments_freq_saves_files, X, Y, alloc, model_name, folder_to_save_validation):
    freq_folder_to_save_validation = os.path.join(folder_to_save_validation, "freq_saves")
    os.makedirs(freq_folder_to_save_validation, exist_ok=True)
    df_epocas_exp = pd.DataFrame()
    for epoch_model in model_experiments_freq_saves_files:
        if "Cluster" in epoch_model:
            return


        model_keras = keras.models.load_model(epoch_model, safe_mode=False,
                                    compile=False)

        epoch = os.path.basename(epoch_model).replace(".keras", "")

        epocs_model_name = f"{model_name}{epoch}epc"
        predictions = model_keras.predict(X)
        predict_score = prediction_score(Y, predictions, alloc, epocs_model_name)
        model_test_filename = os.path.join(freq_folder_to_save_validation, f"{epocs_model_name}_test.npz")
        model_score_filename = os.path.join(freq_folder_to_save_validation, f"{epocs_model_name}_score.json")
        predict_score["epoca"] = epoch

        save_scores(Y, predictions, alloc, model_test_filename, predict_score, model_score_filename)

        if not isinstance(predict_score, list):
            predict_score = [predict_score]
        df_epocas_exp = pd.concat([df_epocas_exp, pd.DataFrame(predict_score)], ignore_index=True)
        df_epocas_exp.reset_index(drop=True, inplace=True)


    df_epocas_exp.columns = df_epocas_exp.columns.str.replace('_', ' ')
    if "name" in df_epocas_exp:
        df_epocas_exp.sort_values(by="name", inplace=True)
        df_epocas_exp["name"] = df_epocas_exp["name"].str.replace('_', ' ')
    if "epoca" in df_epocas_exp:
        df_epocas_exp.sort_values(by="epoca", inplace=True)

    path_schema_csv = os.path.join(freq_folder_to_save_validation, "experiment_results.csv")
    path_schema_tex = os.path.join(freq_folder_to_save_validation, "experiment_results.tex")


    if len(df_epocas_exp)>10:
        df_schema_sort = df_epocas_exp.sort_values(by="optimal percentage", ascending=False)
        df_schema_sort = pd.concat([df_schema_sort.head(), df_schema_sort.tail()])
        df_schema_sort.to_csv(path_schema_csv, index=False)
        df_schema_sort.to_latex(path_schema_tex, escape=False,index=False, float_format="%.2f")
        path_schema_csv = os.path.join(freq_folder_to_save_validation, "experiment_results_complete.csv")
        path_schema_tex = os.path.join(freq_folder_to_save_validation, "experiment_results_complete.tex")

    df_epocas_exp.to_csv(path_schema_csv, index=False)
    df_epocas_exp.to_latex(path_schema_tex, escape=False,index=False, float_format="%.2f")


    return predict_score



for schema in schema_list:
    if schema not in schema_to_validate:
        continue
    # if "linear_weights" not in schema:
    #     continue
    get_dataset, prepare_for_data, prepare_for_model, prediction_from_model, merge_predictions = get_functions_from_experiment(path_to_trained_models_folder, schema)

    df_schema = pd.DataFrame()

    schema_path = os.path.join(path_to_trained_models_folder, schema)
    model_experiments = [filename for filename in os.listdir(schema_path) if os.path.isdir(os.path.join(schema_path,filename))]
    model_experiments = [f for f in model_experiments if "pycache" not in f]
    for model_experiment in model_experiments:
        model_experiment_path = os.path.join(schema_path, model_experiment)
        model_experiment_files = [filename for filename in os.listdir(model_experiment_path) if not os.path.isdir(os.path.join(model_experiment_path,filename))]
        if len(model_experiment_files)==0:
            continue
        model_keras_experiment_path = [f for f in model_experiment_files if f.endswith(".keras")][0]
        model_name = model_keras_experiment_path.replace(".keras", "")
        model_keras_experiment_path = os.path.join(model_experiment_path, model_keras_experiment_path)

        folder_to_save_validation = os.path.join(path_to_validation_folder, schema, model_experiment)
        os.makedirs(folder_to_save_validation, exist_ok=True)


        model_keras = keras.models.load_model(model_keras_experiment_path, safe_mode=False,
                                    compile=False)

        # if isinstance(model_keras, tfdf.keras.RandomForestModel):
        #     model_input_shape = None
        #     model_output_shape = None
        #     continue

        # else:
        model_input_shape = model_keras.input_shape
        model_output_shape = model_keras.output_shape

        extra_args = {}
        data_metadata = prepare_for_data(model_input_shape, model_output_shape)

        X_timeseries = data_metadata["X_timeseries"]
        Y_timeseries = data_metadata["Y_timeseries"]
        frac = 1
        train_features_folga = 24
        skiping_step=Y_timeseries
        if "Y_label_dim" in data_metadata:
            y_val = columns_Y[0]
            num_classes = data_metadata["Y_classes"]-1
            key_g = f"{y_val}_{num_classes}"
            dict_classe[key_g][0]=[0]
            dict_class_to_use = dict_classe[key_g]
            extra_args["classes_dict"]=dict_classe[key_g]
        data_to_go = dataset.copy()
        get_dataset_output = get_dataset(
                data_to_go,
                drop_cols=datetime_col,
                y_columns=columns_Y,
                time_moving_window_size_X=X_timeseries,
                time_moving_window_size_Y=Y_timeseries,
                frac=frac,
                keep_y_on_x=True,  # after some intial anlysis keeping the band of the day before helped the models
                train_features_folga=train_features_folga,        
                skiping_step=skiping_step,
                time_cols=time_cols,
                **extra_args,
            )
        get_dataset_args={

            "y_columns":columns_Y,
            "time_moving_window_size_X":X_timeseries,
            "time_moving_window_size_Y":Y_timeseries,
            "frac":frac,
            "keep_y_on_x":True,
            "train_features_folga":train_features_folga,        
            "skiping_step":skiping_step,
            "time_cols":time_cols,
            "alloc_column":alloc_column,


        }



        X, Y = prepare_for_model(get_dataset_output, model_input_shape, model_output_shape)


        predictions = model_keras.predict(X)

        pred_dict = prediction_from_model(predictions, model_output_shape, model_name, data_metadata)
        test_dataset_Y = Y



        get_dataset_args["y_columns"] = alloc_column

        test_allocation = get_dataset(data_to_go,**get_dataset_args,
        **extra_args,)
        test_allocation = test_allocation[1]


        model_test_filename = os.path.join(folder_to_save_validation, f"{model_name}_test.npz")
        model_score_filename = os.path.join(folder_to_save_validation, f"{model_name}_score.json")
        if "Y_label_dim" in data_metadata:
            test_allocation_labels = assign_labels_with_limits(test_allocation.ravel(), dict_class_to_use)
            test_allocation_labels = np.array(test_allocation_labels).reshape(test_dataset_Y[1].shape)
            reshape_shape2 = test_allocation_labels.shape[:2]
            if isinstance(predictions, list):
                predict_score = prediction_score(test_dataset_Y[0].reshape(reshape_shape2), 
                predictions[0].reshape(reshape_shape2), test_allocation.reshape(reshape_shape2), f"{model_name}_arch")
                acc_score = classificaion_scores(test_dataset_Y[1], np.argmax(predictions[1], axis=2), test_allocation_labels)
                predict_score.update(acc_score)
                if len(predictions)==3:
                    predict_score_inter = prediction_score(test_dataset_Y[2], predictions[2], test_allocation, f"{model_name}_interpr")
                    predict_score=[predict_score, predict_score_inter]
        else:
            predict_score = prediction_score(test_dataset_Y, predictions, test_allocation, model_name)
            save_scores(test_dataset_Y, predictions, test_allocation, model_test_filename, predict_score, model_score_filename)

        if not isinstance(predict_score, list):
            predict_score = [predict_score]
        df_schema = pd.concat([df_schema, pd.DataFrame(predict_score)], ignore_index=True)
        df_schema.reset_index(drop=True, inplace=True)


        # Frequent saves
        model_experiments_freq_saves = [filename for filename in os.listdir(model_experiment_path) if os.path.isdir(os.path.join(model_experiment_path,filename))]
        model_experiments_freq_saves_path = os.path.join(model_experiment_path, model_experiments_freq_saves[0])
        model_experiments_freq_saves_files = [ os.path.join(model_experiments_freq_saves_path, filename) for filename in os.listdir(model_experiments_freq_saves_path) if not filename.startswith("unfinished")]
        make_validation_epoch_scores(model_experiments_freq_saves_files, X, Y, test_allocation, model_name, folder_to_save_validation)


    df_schema.columns = df_schema.columns.str.replace('_', ' ')
    if "name" in df_schema:
        df_schema.sort_values(by="name", inplace=True)
        df_schema["name"] = df_schema["name"].str.replace('_', ' ')
        
    path_schema_csv = os.path.join(path_to_validation_folder, schema, "experiment_results.csv")
    path_schema_tex = os.path.join(path_to_validation_folder, schema, "experiment_results.tex")


    if len(df_schema)>10:
        df_schema_sort = df_schema.sort_values(by="optimal percentage", ascending=False)
        df_schema_sort = pd.concat([df_schema_sort.head(), df_schema_sort.tail()])
        df_schema_sort.to_csv(path_schema_csv, index=False)
        df_schema_sort.to_latex(path_schema_tex, escape=False,index=False, float_format="%.2f")
        path_schema_csv = os.path.join(path_to_validation_folder, schema, "experiment_results_complete.csv")
        path_schema_tex = os.path.join(path_to_validation_folder, schema, "experiment_results_complete.tex")

    df_schema.to_csv(path_schema_csv, index=False)
    df_schema.to_latex(path_schema_tex, escape=False,index=False, float_format="%.2f")

