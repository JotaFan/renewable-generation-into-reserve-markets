import numpy as np

def metric_scores_renewtese(name, predictions, truth_data, benchmark=False):
    erro = truth_data - predictions
    erro_abs = np.abs(erro)
    erro_abs_sum = np.nansum(erro_abs)

    mse = np.nanmean(np.square(erro))
    rmse = np.sqrt(mse)

    erro_benchmark = truth_data - benchmark
    erro_benchmark_abs = np.abs(erro_benchmark)
    erro_benchmark_abs_sum = np.nansum(erro_benchmark_abs)

    erro_benchmark_mse = np.nanmean(np.square(erro_benchmark))
    erro_benchmark_rmse = np.sqrt(erro_benchmark_mse)

    surplus_mask = erro<0
    missing_mask = erro>0
    surplus_mask_benchmark = erro_benchmark<0
    missing_mask_benchmark = erro_benchmark>0

    abs_better_than_benchmark_mask = erro_benchmark_abs<erro_abs


    alloc_missing = np.where(
        predictions > truth_data, 0, truth_data - predictions
    )
    alloc_surplus = np.where(
        predictions < truth_data, 0, predictions - truth_data
    )

    benchmark_alloc_missing = np.where(
        benchmark > truth_data, 0, truth_data - benchmark
    )
    benchmark_alloc_surplus = np.where(
        benchmark < truth_data, 0, benchmark - truth_data
    )

    smaller_error = erro_abs < erro_benchmark_abs

    mask_great_or_equal = predictions >= truth_data

    optimal_mask = mask_great_or_equal & smaller_error
    optimal_percentage = (np.sum(optimal_mask) / truth_data.size) * 100

    benchmark_alloc_missing_sum = np.sum(benchmark_alloc_missing)
    benchmark_alloc_surplus_sum = np.sum(benchmark_alloc_surplus)

    alloc_missing_sum = np.sum(alloc_missing)
    alloc_surplus_sum = np.sum(alloc_surplus)

    GPD_total = ((erro_benchmark_abs_sum - erro_abs_sum) / (erro_benchmark_abs_sum)) * 100
    GPD_F_total = ((benchmark_alloc_missing_sum - alloc_missing_sum) / (benchmark_alloc_missing_sum)) * 100
    GPD_D_total = ((benchmark_alloc_surplus_sum - alloc_surplus_sum) / (benchmark_alloc_surplus_sum)) * 100
    GPD_norm_total = np.mean([GPD_D_total, GPD_F_total])

    missing_smaller = alloc_missing_sum < benchmark_alloc_missing_sum
    surplus_smaller = alloc_surplus_sum < benchmark_alloc_surplus_sum

    better_than_benchmark = missing_smaller & surplus_smaller

    if better_than_benchmark:
        GPD_positivo_total = GPD_total
    else:
        GPD_positivo_total = 0

    GPD_D_form2_total = GPD_D_total
    GPD_F_form2_total = GPD_F_total
    if GPD_D_form2_total < 0:
        GPD_D_form2_total = (GPD_D_form2_total**2)*-1
    if GPD_F_form2_total < 0:
        GPD_F_form2_total = (GPD_F_form2_total**2)*-1
    GPD_norm2_total = np.mean([GPD_D_form2_total, GPD_F_form2_total])



    delta_erros = erro_benchmark - erro
    sum_erros = erro_benchmark + erro
    GPD = ((erro_benchmark_abs - erro_abs)/(erro_benchmark_abs))*100

    GPD_D= np.full(truth_data.shape, np.nan)
    GPD_F= np.full(truth_data.shape, np.nan)
    zero_mask = erro==0
    GPD_D[zero_mask] = 100
    GPD_F[zero_mask] = 100




    GPD_D[surplus_mask_benchmark] = GPD[surplus_mask_benchmark]
    GPD_F[missing_mask_benchmark] = GPD[missing_mask_benchmark]

    GPD_norm = np.nanmean([GPD_D, GPD_F], axis=0)

    GPD_D_fornorm2 = GPD_D.copy()
    GPD_F_fornorm2 = GPD_F.copy()
    GPD_D_fornorm2[GPD_D_fornorm2<0]=((GPD_D_fornorm2[GPD_D_fornorm2<0]-1)**2)*-1
    GPD_F_fornorm2[GPD_F_fornorm2<0]=((GPD_F_fornorm2[GPD_F_fornorm2<0]-1)**2)*-1
    GPD_norm2 = np.nanmean([GPD_D_fornorm2, GPD_F_fornorm2], axis=0)

    better_when_surplus = GPD_D>0
    better_when_missing = GPD_F>0
    both_better = better_when_surplus & better_when_missing

    GPD_positivo = GPD.copy()
    GPD_positivo[~both_better]=0

    GPD_all_positive=0
    GPD_all_positive_total=0

    if GPD_positivo_total>0:
        if np.nanmean(GPD_positivo)>0:
            if np.nanmean(GPD_D)>0:
                if np.nanmean(GPD_F)>0:
                    GPD_all_positive = np.nanmean(GPD_positivo)
                    GPD_all_positive_total=GPD_positivo_total


    predict_score = {
        "name": [name],
        "RMSE": [rmse],
        "SAE": [erro_abs_sum],
        "AllocF": [alloc_missing_sum],
        "AllocD": [alloc_surplus_sum],



        "GPD F": [GPD_F_total],
        "GPD D": [GPD_D_total],
        "GPD": [GPD_total],
        "GPD norm": [GPD_norm_total],
        "GPD Positivo": [GPD_positivo_total],
        "GPD norm2": [GPD_norm2_total],
        "OptPer": [optimal_percentage],
        "benchmark SAE":[erro_benchmark_abs_sum],
        "benchmark rmse":[erro_benchmark_rmse],
        "benchmark AllocF":[benchmark_alloc_missing_sum],
        "benchmark AllocD":[benchmark_alloc_surplus_sum],




        "GPD medio":[np.nanmean(GPD)],
        "GPD D medio":[np.nanmean(GPD_D)],
        "GPD F medio":[np.nanmean(GPD_F)],
        "GPD norm medio":[np.nanmean(GPD_norm)],
        "GPD norm2 medio":[np.nanmean(GPD_norm2)],
        "GPD Positivo medio":[np.nanmean(GPD_positivo)],
        "GPD_all_positive":[GPD_all_positive],
        "GPD_all_positive_total":[GPD_all_positive_total],


    }

    return predict_score
