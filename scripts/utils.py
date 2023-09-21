
import math
import os
import pandas as pd
from keras.callbacks import Callback, ModelCheckpoint

STEPS_PER_EPOCH = 2336

class CustomModelCheckpoint(Callback):
    def __init__(self, model_keras_filename, period, start_epoch=1):
        super().__init__()
        self.model_keras_filename = model_keras_filename
        self.period = period
        self.start_epoch = start_epoch
        self.model_checkpoint = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.model_checkpoint is not None:
            self.model_checkpoint.model = None

        if epoch >= self.start_epoch:
            self.model_checkpoint = ModelCheckpoint(self.model_keras_filename,
                                                     save_freq=int(self.period * STEPS_PER_EPOCH),
                                                     )
            self.model_checkpoint.set_model(self.model)

    def on_epoch_end(self, epoch, logs=None):
        if self.model_checkpoint is not None:
            self.model_checkpoint.on_epoch_end(epoch, logs)

class StopOnNanLoss(Callback):
    def __init__(self, filepath):
        super(StopOnNanLoss, self).__init__()
        self.filepath = filepath
        self.last_good_model = None
        self.last_good_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        self.last_good_model = self.model.get_weights()
        self.last_good_epoch = epoch


    def on_train_batch_end(self, batch, logs=None):
        loss = logs.get('loss')
        if loss is not None and (isinstance(loss, float) and math.isnan(loss)):
            print(f"Stopping training due to NaN loss at batch {batch}.")
            if self.last_good_model is not None:
                self.model.set_weights(self.last_good_model)
            if self.last_good_epoch is not None:
                frq_model_filename = self.filepath.replace(".keras", f"freq_saves/{self.last_good_epoch}.keras")
                self.model.save(frq_model_filename)
            self.model.save(frq_model_filename)
            self.model.stop_training = True
            self.model.save(self.filepath.replace(".keras", "freq_saves/unfinished.keras"))
        else:
            self.last_good_model = self.model.get_weights()
            if os.path.exists(self.filepath.replace(".keras", "freq_saves/unfinished.keras")):
                os.remove(self.filepath.replace(".keras", "freq_saves/unfinished.keras"))


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


