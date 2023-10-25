
import math
import os
import pandas as pd
from keras_core.callbacks import Callback, ModelCheckpoint
import json
STEPS_PER_EPOCH = 2336

def update_history_dict(new_log, old_log):
    history_to_save = {}
    if old_log:
        if isinstance(old_log, dict):
            for key in old_log:
                old = old_log[key]
                new = new_log.get(key, [])
                if not isinstance(new, list):
                    new=[new]
                if not isinstance(old, list):
                    old=[old]
                old = [f for f in old if f]
                old = [f for f in old if not  math.isnan(f)]
                history_to_save[key] = old + new
    for key in new_log:
        if key not in history_to_save:
            history_to_save[key] = new_log[key]
    return history_to_save



class SaveModelCallback(Callback):
    def __init__(self, save_frequency, model_keras_filename,model_log_filename, logs=None,start_epoch=0 ):
        super(SaveModelCallback, self).__init__()
        self.save_frequency = save_frequency
        self.model_keras_filename = model_keras_filename
        self.model_log_filename = model_log_filename
        self.start_epoch = start_epoch
        self.logs = logs or {}


    def on_epoch_end(self, epoch, logs=None):
        epoc_save = epoch + 1 + self.start_epoch
        self.logs = update_history_dict(logs, self.logs)
        if (epoc_save) % self.save_frequency == 0:
            model_save_name = self.model_keras_filename.format(epoch=epoc_save)
            # Save the model
            self.model.save(model_save_name)

            # Save the logs to a JSON file
            with open(self.model_log_filename, "w") as f:
                json.dump(self.logs, f)



class StopOnNanLoss(Callback):
    def __init__(self, filepath, model_log_filename, logs=None,save_frequency=1, model_keras_filename=None, start_epoch=0):
        super(StopOnNanLoss, self).__init__()
        self.filepath = filepath
        self.last_good_model = None
        self.last_good_epoch = None
        self.logs = logs or {}
        self.model_log_filename = model_log_filename
        self.model_keras_filename = model_keras_filename or filepath
        self.start_epoch = start_epoch
        self.save_frequency = save_frequency


    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        nan_value = math.isnan(loss) | math.isinf(loss)
        if loss is not None and (isinstance(loss, float) and not nan_value):
            self.last_good_model = self.model.get_weights()
            self.last_good_epoch = epoch
            self.logs = update_history_dict(logs, self.logs)
            epoc_save = epoch + 1 + self.start_epoch
            if (epoc_save) % self.save_frequency == 0:
                model_save_name = self.model_keras_filename.format(epoch=epoc_save)
                # Save the model
                self.model.save(model_save_name)

                # Save the logs to a JSON file
                with open(self.model_log_filename, "w") as f:
                    json.dump(self.logs, f)

    def on_train_batch_end(self, batch, logs=None):
        loss = logs.get('loss')
        nan_value = math.isnan(loss) | math.isinf(loss)
        if loss is not None and (isinstance(loss, float) and nan_value):
            print(f"Stopping training due to NaN loss at batch {batch}.")
            if self.last_good_model is not None:
                self.model.set_weights(self.last_good_model)
            if self.last_good_epoch is not None:
                frq_model_filename = self.filepath.replace(".keras", f"freq_saves/{self.last_good_epoch}.keras")
                self.model.save(frq_model_filename)
                with open(self.model_log_filename, "w") as f:
                    json.dump(self.logs, f)
            self.model.save(self.filepath.replace(".keras", "freq_saves/unfinished.keras"))
            self.model.stop_training = True
        else:
            self.last_good_model = self.model.get_weights()
            unfinished = self.filepath.replace(".keras", "freq_saves/unfinished.keras")
            if os.path.exists(unfinished):
                os.remove(unfinished)


def assign_labels_with_limits(values, classes_dict):
    labels = []
    last_label = sorted([int(f) for f in classes_dict.keys()])[-1]
    for value in values:
        for label, limits in classes_dict.items():
            label = int(label)
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

    return labels

import numpy as np


def moving_mean_with_step(prediction, timesteps, skip_step=1):
    if not skip_step:
        skip_step =1

    for i in range(timesteps*skip_step):
        prediction = np.vstack([prediction,np.full(timesteps, np.nan)])


    df = pd.DataFrame(prediction)

    shidt_count = 0 
    for col in df.columns:
        df[col] = df[col].shift(shidt_count*skip_step)
        shidt_count+=1

    df.dropna(how="all", inplace=True)

    return df.mean(axis=1).to_numpy()


def moving_mean_predictions(prediction, skip_step=1):
    prediction = np.array(prediction)
    if len(prediction.shape)==2:
        batch, timesteps = prediction.shape
        classes=1
    elif len(prediction.shape)==3: # LETS assume it is ==3
        batch, timesteps, classes = prediction.shape
        if classes ==1:
            prediction = prediction[:,:,0]

    try:
        pass
    except:
        print("wtf tis dfddd")
        print(prediction.shape)

    moving_mean_stack = None
    for i in range(classes):
        moving_mean = moving_mean_with_step(prediction, timesteps, skip_step)
        if not moving_mean_stack:
            moving_mean_stack =  moving_mean      
        else:
            moving_mean_stack =  np.stack([moving_mean_stack,moving_mean], axis=1)      
    return moving_mean_stack


