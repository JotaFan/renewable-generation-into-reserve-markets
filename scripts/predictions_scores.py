
from sklearn.metrics import r2_score
import numpy as np
import json

def prediction_score(test_dataset_Y, predictions, test_allocation, model_name):

    erro = test_dataset_Y - predictions
    erro_abs = np.nansum(np.abs(erro))
    r2 = r2_score(np.nan_to_num(test_dataset_Y.ravel()), np.nan_to_num(predictions.ravel()))
    
    # Test has both zeros and nan, but no neg values, so we shift by one
    percentage_error = ((test_dataset_Y - predictions) +1)/(test_dataset_Y+1)
    mape = np.nanmean(np.abs(percentage_error))

    mse= np.nanmean(np.square(erro))
    rmse= np.sqrt(mse)

    nrmse_mean = rmse / np.nanmean(test_dataset_Y)
    nrmse_spread = rmse / (np.max(test_dataset_Y) - np.min(test_dataset_Y)) 


    erro_spain = test_dataset_Y - test_allocation
    erro_pred = test_dataset_Y - predictions

    np.abs(erro_spain)  
    np.abs(erro_pred)

    # Alocacao em falta, e alocaçao a mais
    alloc_missing = np.where(predictions >= test_dataset_Y, 0, test_dataset_Y-predictions)
    alloc_surplus = np.where(predictions < test_dataset_Y, 0, predictions-test_dataset_Y)

    np.where(test_allocation >= test_dataset_Y, 0, test_dataset_Y-test_allocation)
    np.where(test_allocation < test_dataset_Y, 0, predictions-test_allocation)

    # Percentagem das vezes que o modelo é melhor que o espanhol
    # Cenario optimo
    # maior ou igual a test, e menor que allocation
    mask_great_or_equal = predictions >= test_dataset_Y
    mask_smaller_or_equal_spain = predictions <= test_allocation
    optimal_mask = mask_great_or_equal & mask_smaller_or_equal_spain

    predict_score = {
        "name":model_name,
        "rmse":rmse,
        "Absolute Error": erro_abs, 
        "r2 score":r2,
        "mape score": mape, 
        "alloc_missing":np.sum(alloc_missing),
        "alloc_surplus":np.sum(alloc_surplus),
        "spain_alloc_missing":np.sum(alloc_missing),
        "spain_alloc_surplus":np.sum(alloc_missing),
        "optimal_percentage":np.sum(optimal_mask)/len(optimal_mask),
        
      }



    return predict_score

def save_scores(test_dataset_Y, predictions, test_allocation, model_test_filename, predict_score, model_score_filename):
    pred_dict = {"test":test_dataset_Y,
                "prediction":predictions,
                "test_allocation":test_allocation}
    np.savez_compressed(model_test_filename, **pred_dict)

    with open(model_score_filename, "w") as mfile:
        json.dump(predict_score, mfile)

    return