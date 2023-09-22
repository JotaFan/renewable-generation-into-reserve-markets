
from sklearn.metrics import r2_score
import numpy as np
import json

def prediction_score(test_dataset_Y, predictions, test_allocation, model_name):

    erro = test_dataset_Y - predictions
    erro_abs = np.abs(erro)
    erro_abs_sum = np.nansum(erro_abs)
    r2 = r2_score(np.nan_to_num(test_dataset_Y.ravel()), np.nan_to_num(predictions.ravel()))
    
    # Test has both zeros and nan, but no neg values, so we shift by one
    percentage_error = ((test_dataset_Y - predictions) +1)/(test_dataset_Y+1)
    mape = np.nanmean(np.abs(percentage_error))

    mse= np.nanmean(np.square(erro))
    rmse= np.sqrt(mse)

    nrmse_mean = rmse / np.nanmean(test_dataset_Y)
    nrmse_spread = rmse / (np.max(test_dataset_Y) - np.min(test_dataset_Y)) 


    erro_spain = test_dataset_Y - test_allocation
    erro_spain_abs = np.abs(erro_spain)
    erro_spain_abs_sum = np.nansum(erro_spain_abs)


    # Alocacao em falta, e alocaçao a mais
    alloc_missing = np.where(predictions >= test_dataset_Y, 0, test_dataset_Y-predictions)
    alloc_surplus = np.where(predictions < test_dataset_Y, 0, predictions-test_dataset_Y)

    spain_alloc_missing = np.where(test_allocation >= test_dataset_Y, 0, test_dataset_Y-test_allocation)
    spain_alloc_surplus = np.where(test_allocation < test_dataset_Y, 0, test_allocation-test_dataset_Y)

    # Percentagem das vezes que o modelo é melhor que o espanhol
    # Cenario optimo
    # maior ou igual a test, e menor que allocation

    mask_great_or_equal_spain = test_allocation >= test_dataset_Y
    mask_smaller_or_equal_model = test_allocation <= predictions
    anti_optimal_mask = mask_great_or_equal_spain & mask_smaller_or_equal_model
    anti_optimal_percentage = np.sum(anti_optimal_mask)/test_dataset_Y.size


    # Melhor
    # optimo + aqueles que estao mais perto (erro mais pequeno)
    smaller_error = erro_abs <= erro_spain_abs

    # mais que test (so quando tambem menos que alocado)
    mask_great_or_equal = predictions >= test_dataset_Y
    mask_smaller_or_equal_spain = predictions <= test_allocation

    better_allocation_mask = mask_great_or_equal & mask_smaller_or_equal_spain
    better_allocation_mask = np.sum(better_allocation_mask)/test_dataset_Y.size


    optimal_mask = mask_great_or_equal & smaller_error
    optimal_percentage = np.sum(optimal_mask)/test_dataset_Y.size

    both_alocated = mask_great_or_equal_spain & mask_great_or_equal

 
    # Assumir que é prioridade ter alocado, meljor que espanha é erro menor e ter alocado,
    # better_than_spain = smaller_error & mask_great_or_equal # assim teriamos de assumir que se eu alocasse 100000000 para 100 e espanha 95 que o meu era melhor..
    beter_percentage = np.sum(smaller_error)/test_dataset_Y.size

        # "spain alloc missing":np.sum(spain_alloc_missing),
        # "spain alloc surplus":np.sum(spain_alloc_surplus),


    predict_score = {
        "name":model_name,
        "rmse":rmse,
        "abs erro": erro_abs_sum,
        "erro comp": str(erro_abs_sum<erro_spain_abs_sum),
        "r2 score":r2,
        "mape score": mape, 
        "alloc missing":np.sum(alloc_missing),
        "alloc surplus":np.sum(alloc_surplus),
        "optimal percentage":optimal_percentage*100,
        "better allocation": better_allocation_mask*100,
        "beter percentage": beter_percentage*100,

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