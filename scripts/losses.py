
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import (MeanAbsoluteError,
                                     MeanSquaredLogarithmicError)

OFFSET=30


def weighted_loss(y_true, y_pred):
    weights = tf.abs(y_true - y_pred)  # Calculate weights based on the absolute difference between true and predicted values
    loss = tf.square(y_true - y_pred)  # Calculate the squared loss
    weighted_loss = tf.multiply(weights, loss)  # Multiply the weights with the loss
    return tf.reduce_mean(weighted_loss)  # Calculate the mean of the weighted loss

def mean_squared_diff_error(y_true, y_pred):
    return K.mean(K.abs(K.square(y_pred) - K.square(y_true)))

def mean_squared_diff_log_error(y_true, y_pred):
    return K.mean(K.abs(K.log(K.square(y_true)+1) - K.log(K.square(y_pred)+1)))


def mean_gamma_deviance(y_true, y_pred):
    return tf.reduce_mean(tf.math.pow(y_true - y_pred, 2) / tf.math.pow(y_true, 2))


def d2_absolute_error_score(y_true, y_pred):
    return tf.reduce_mean(tf.math.abs(y_true - y_pred) / (tf.math.abs(y_true) + tf.math.abs(y_pred)))

def d2_pinball_score(y_true, y_pred):
    error = y_pred - y_true
    positive_error = tf.maximum(0., error)
    negative_error = tf.maximum(0., -error)
    return tf.reduce_mean(positive_error * 0.9 + negative_error * 0.1)


def d2_tweedie_score(y_true, y_pred):
    p = 1.5  # Tweedie power parameter
    return tf.reduce_mean(tf.pow(tf.maximum(0., y_true), 2-p) * tf.pow(tf.maximum(0., y_pred), p-1) - y_true * y_pred)

def mean_poisson_deviance(y_true, y_pred):
    return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred))


# dont use MAPE because of zeros in Y
def mean_absolute_percentage_error(y_true, y_pred):
    return (K.abs(y_pred - y_true)+OFFSET)/(y_true+OFFSET)


def root_squared_error(y_true, y_pred):
    return K.sqrt(K.square(y_pred - y_true))


# def root_mean_squared_log_error(y_true, y_pred):
#     msle = MeanSquaredLogarithmicError()
#     return K.sqrt(msle(y_true, y_pred)) 

# def root_mean_squared_error(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true)))


def relative_root_mean_squared_error(y_true, y_pred):
    rmse = K.sqrt(K.mean(K.square(y_pred - y_true)))
    mean = K.mean(y_true)
    return rmse / mean


def nan_mean_squared_error_loss(nan_value=np.nan):
    # Create a loss function
    def loss(y_true, y_pred):
        # if y_true.shape != y_pred.shape:
        #    y_true = y_true[:, :1]
        indices = tf.where(tf.not_equal(y_true, nan_value))
        return tf.keras.losses.mean_squared_error(
            tf.gather_nd(y_true, indices), tf.gather_nd(y_pred, indices)
        )

    # Return a function
    return loss

def root_mean_squared_error():
    # Create a loss function
    def loss(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


    # Return a function
    return loss

def mean_squared_log_error():
    # Create a loss function
    def loss(y_true, y_pred):
        msle = MeanSquaredLogarithmicError()
        return msle(y_true, y_pred)


    # Return a function
    return loss


def nan_root_mean_squared_log_error(y_true, y_pred):
    # Create a loss function
    msle = MeanSquaredLogarithmicError()
    mask = tf.math.is_finite(y_true) & tf.math.is_finite(y_pred)

    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    # Return a function
    return K.sqrt(msle(y_true, y_pred)) 


losses_dict = {"mae":MeanAbsoluteError, "msle":MeanSquaredLogarithmicError, 
               "rmse":root_mean_squared_error,
               "rmsle":mean_squared_log_error}
