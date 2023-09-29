# Generator for models with linear predicctions, is:
# output shape:
# (N, y_time, Y_features)
# input shape:
# (N, x_time, n_features)

import os
import sys


top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(top_level_dir)

from scripts.generator import DataGenerator, get_dataset


def prepare_for_data(model_input, model_output):
    data_metadata = {
        "multioutput":False,
        "multinput":False,

    }


    if isinstance(model_input, list):
        data_metadata["multinput"]=True
        data_metadata["X_timeseries"] = model_input[0][1]
        data_metadata["X_features"] = len(model_input)

    else:
        data_metadata["X_timeseries"] = model_input[1]
        data_metadata["X_features"] = model_input[2]

    if isinstance(model_output, list):
        data_metadata["multioutput"]=True
        data_metadata["Y_timeseries"] = model_output[0][1]
        data_metadata["Y_features"] = len(model_output)
    else:
        data_metadata["Y_timeseries"] = model_output[1]
        data_metadata["Y_features"] = model_output[2]


    

    return data_metadata

def prepare_for_model(get_dataset_output, model_input, model_output):
    train_dataset_X, train_dataset_Y, test_dataset_X, test_dataset_Y, gen = get_dataset_output

    
    return train_dataset_X, train_dataset_Y

def prediction_from_model(predictions, model_output, model_name, data_metadata):
    pred_dict = {}
    pred_dict[f"{model_name}_prediction"] = predictions.ravel()


    return pred_dict
