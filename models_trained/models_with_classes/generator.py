# Generator for models that have cluster predictions, is:
# output shape:
#   [(N, y_time, 1), (N, y_time, n_classes), (N, y_time, 1)] or just the 1st two.
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


from scripts.utils import assign_labels_with_limits

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
        classes_dict=None,
        label_col=None,
    ):
        self.train_features_folga=train_features_folga
        self.skiping_step = skiping_step
        self.classes_dict = classes_dict

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
        labels = assign_labels_with_limits(Y, self.classes_dict)
        
        return X, Y, labels

        



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
                        classes_dict = {},
                        label_col = "labels",



                
):
    if not label_col in dataset:
        print(len(dataset))
        print(len(dataset[y_columns].values.ravel()))
        print(len(assign_labels_with_limits(dataset[y_columns].values.ravel(), classes_dict)))
        print(classes_dict)
        dataset[label_col] = assign_labels_with_limits(dataset[y_columns].values.ravel(), classes_dict)
    gen = DataGenerator(dataset, time_moving_window_size_X, time_moving_window_size_Y, y_columns, keep_y_on_x, 
                        drop_cols=drop_cols,
                                                train_features_folga=train_features_folga,
                                                skiping_step=skiping_step,
                                                time_cols=time_cols,
                                                        phased_out_columns = phased_out_columns, 
                                                        classes_dict=classes_dict,
                                                        label_col=label_col,)


    X, Y, labels = [], [], []
    for x, y, l in gen:
        X.append(x)
        Y.append(y)
        labels.append(l)
    X = np.array(X)
    Y = np.array(Y)
    labels = np.array(labels)

    train_len = math.ceil(frac * len(X))
    test_len = len(X) - train_len
    
    
    train_dataset_X = X[:train_len]
    test_dataset_X = X[train_len:train_len+test_len]


    train_dataset_Y = Y[:train_len]
    test_dataset_Y = Y[train_len:train_len+test_len]
    
    train_dataset_labels = labels[:train_len]
    test_dataset_labels = labels[train_len:train_len+test_len]
    
    
    return train_dataset_X, train_dataset_Y, test_dataset_X, test_dataset_Y, train_dataset_labels, test_dataset_labels


def prepare_for_data(model_input, model_output):
    data_metadata = {
        "multioutput":False,

    }
    if isinstance(model_output, list):
        data_metadata["multioutput"] = True
        data_metadata["output_len"] = len(model_output)
        if data_metadata["output_len"]==3:
            data_metadata["Y_timeseries"] = model_output[0][1]
            data_metadata["Y_classes"] = model_output[1][2]
            data_metadata["Y_features"] = model_output[0][2]
            data_metadata["Y_label_dim"] = 1
        if data_metadata["output_len"]==2:
            if model_output[0][2] > model_output[1][2]:
                data_metadata["Y_label_dim"] = 0
                data_metadata["Y_linear_dim"] = 1
            else:
                data_metadata["Y_label_dim"] = 1
                data_metadata["Y_linear_dim"] = 0

            data_metadata["Y_timeseries"] = model_output[data_metadata["Y_label_dim"]][1]
            data_metadata["Y_classes"] = model_output[data_metadata["Y_label_dim"]][2]
            data_metadata["Y_features"] = model_output[data_metadata["Y_linear_dim"]][2]

    data_metadata["X_timeseries"] = model_input[1]
    data_metadata["X_features"] = model_input[2]
    

    return data_metadata

def prepare_for_model(get_dataset_output, model_input, model_output):
    train_dataset_X, train_dataset_Y, test_dataset_X, test_dataset_Y, train_dataset_labels, test_dataset_labels = get_dataset_output

    if isinstance(model_output, list):
        if len(model_output)==3:
            Y = [train_dataset_Y, train_dataset_labels, train_dataset_Y]
            X = train_dataset_X
        elif len(model_output)==2:
            Y = [train_dataset_labels, train_dataset_Y]
            X = train_dataset_X


    
    return X, Y

def prediction_from_model(predictions, model_output, model_name, data_metadata):
    Y_unet = None
    pred_dict = {}
    if isinstance(model_output, list):
        for i in predictions:
            print(i.shape)
        if len(model_output)==3:
            Y_unet, Y_labels, Y_dense = predictions
        elif len(model_output)==2:
            Y_labels = predictions[data_metadata["Y_label_dim"]]
            Y_dense = predictions[data_metadata["Y_linear_dim"]]
    pred_dict[f"{model_name}_prediction"] = Y_dense.ravel()
    pred_dict[f"{model_name}_prediction_label"] = np.argmax(Y_labels, axis=2).ravel()
    if Y_unet is not None:
        pred_dict[f"{model_name}_Y_unet"] = Y_unet.ravel()


    return pred_dict