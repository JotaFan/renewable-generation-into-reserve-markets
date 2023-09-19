# Generator for models that have cluster predictions, is:
# output shape:
# (N, y_time, Y_features)
# input shape:
# (N, x_time, n_features)

from tensorflow import keras
import math
import pandas as pd
import numpy as np
import math
import os
import sys

import numpy as np

top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(top_level_dir)



class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        dataset:pd.DataFrame,
        time_moving_window_size_X=7*24, #batch size 7 days, 168 hours,
        time_moving_window_size_Y=1*24, #to predict the after 1 day, 24 hours,
        y_columns = [],
        keep_y_on_x=True,
        drop_cols = "datetime",
        train_features_folga=0, # 24 horas de diferenca DA
        skiping_step=1,
        time_cols = [],
        phased_out_columns = [],
    ):
        self.train_features_folga=train_features_folga
        self.skiping_step = skiping_step

        # Make the y the 1st column
        dataset = dataset[y_columns+[col for col in dataset.columns if col not in y_columns]]
        
        # make the time columns the last
        dataset = dataset[[col for col in dataset.columns if col not in time_cols]+time_cols]
        



        if drop_cols:
            dataset = dataset.drop(drop_cols, axis=1)
        if len(phased_out_columns)>0:
            dataset[phased_out_columns] = dataset[phased_out_columns].shift(train_features_folga)
        dataset.dropna(inplace=True)
        self.y_columns = y_columns
        self.y = dataset[self.y_columns].to_numpy()
        if not keep_y_on_x:
            self.x = dataset.loc[:, ~dataset.columns.isin(self.y_columns)].to_numpy()
        else:
            self.x = dataset.to_numpy()
        self.x_batch = time_moving_window_size_X
        self.y_batch = time_moving_window_size_Y
        
        self.dataset_size = len(dataset)
        
    def __len__(self):

        total_batches = self.dataset_size - sum([self.x_batch,self.y_batch]) + 1
        return int(math.ceil(total_batches / self.skiping_step))




    def __getitem__(self, index):
        
        ind = index*self.skiping_step

        limit_point = ind+self.x_batch
        
        X = self.x[ind:limit_point]
        Y = self.y[limit_point:limit_point+self.y_batch]
        
        return X, Y

        



def get_dataset(dataset,  time_moving_window_size_X=168, #batch size 7 days, 168 hours,
        time_moving_window_size_Y=24, #to predict the after 1 day, 24 hours,
        y_columns = [],
        keep_y_on_x=True,
                frac=0.9,
                drop_cols="datetime",
                train_features_folga=0,
                skiping_step=1,
                        time_cols = [],
                        phased_out_columns = ["UpwardUsedSecondaryReserveEnergy", "DownwardUsedSecondaryReserveEnergy"],



                
):

    gen = DataGenerator(dataset, time_moving_window_size_X, time_moving_window_size_Y, y_columns, keep_y_on_x, 
                        drop_cols=drop_cols,
                                                train_features_folga=train_features_folga,
                                                skiping_step=skiping_step,
                                                time_cols=time_cols,
                                                        phased_out_columns = phased_out_columns, 
                                                        )


    X, Y,  =  [], []
    for x, y in gen:
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)

    train_len = math.ceil(frac * len(X))
    test_len = len(X) - train_len
    
    
    train_dataset_X = X[:train_len]
    test_dataset_X = X[train_len:train_len+test_len]


    train_dataset_Y = Y[:train_len]
    test_dataset_Y = Y[train_len:train_len+test_len]
        
    
    return train_dataset_X, train_dataset_Y, test_dataset_X, test_dataset_Y


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
    train_dataset_X, train_dataset_Y, test_dataset_X, test_dataset_Y = get_dataset_output

    
    return train_dataset_X, train_dataset_Y

def prediction_from_model(predictions, model_output, model_name, data_metadata):
    pred_dict = {}
    pred_dict[f"{model_name}_prediction"] = predictions.ravel()


    return pred_dict