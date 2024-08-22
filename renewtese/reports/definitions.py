import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from muaddib._default_functions import prediction_score
from muaddib.shaihulud_utils import load_json_dict

def time_frame_images(year_data, month_data, week_data, day_data, folder_figures=None,name=""):
    import matplotlib.dates as mdates


    # 1 year = 24*365 hours, 1 month = 24*30 hours, 1 week = 24*7 hours, 1 day = 24 hours
    time_ranges = [24*30, 24*7, 24]
    time_ranges_names = ['1 month', '1 week', '1 day']

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    # axs = axs.flatten()
    gs = GridSpec(2, 2, figure=fig)

    axs = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    # Check target distribution
    for ax, time_range, time_range_name in zip(axs, [ month_data, week_data, day_data], time_ranges_names):
        # for count, col in enumerate(columns_Y):
        time_range = time_range[["prediction", "test", "benchmark", 'datetime']]
            
        dataset2 = time_range.set_index('datetime')

        dataset2[["prediction", "test", "benchmark"]].plot(ax=ax)

        ax.set_title(time_range_name)
        #ax.xaxis.set_major_locator(mdates.YearLocator())
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Adjust the vertical spacing here
    plt.subplots_adjust(hspace = 0.5)
    title = name
    fig.suptitle(name)

    figure_name = f"{name}_timeseries_windows.png"
    folder_figures = folder_figures or "."
    #plt.savefig(os.path.join(folder_figures, figure_name), bbox_inches='tight')                                                                                                      

    plt.savefig(os.path.join(folder_figures, figure_name))
    plt.close()
                                                                                                   


def get_prediciton_score(df, metric=""):

    ps = prediction_score(df["test"], df["prediction"], df["benchmark"], "name")[metric]

    return ps

def time_frame_validations(best_result_path, name, name_metric, folder_to_save=None):

    if not isinstance(name_metric, list):
        name_metric=[name_metric]

    best_result = np.load(best_result_path)
    prediction = best_result["prediction"].ravel()
    test_dataset_Y = best_result["test"].ravel()
    test_allocation = best_result["benchmark"].ravel()
    best_result = pd.DataFrame({"prediction":prediction,"test":test_dataset_Y,"benchmark":test_allocation})

    

    num_hours = len(best_result)
    start_date = pd.to_datetime("2019-01-01")
    dates = pd.date_range(start=start_date, periods=num_hours, freq="h")

    best_result["datetime"] = dates
    # get best year
    best_result['year'] = best_result['datetime'].dt.year
    best_result['month'] = best_result['datetime'].dt.month
    best_result['week'] = [f.weekofyear for f in best_result['datetime']]
    best_result['day'] = best_result['datetime'].dt.day

    for met in name_metric:

        best_result_day = best_result.groupby(['year', 'week', 'day']).apply(get_prediciton_score, metric=met)
        best_result_week = best_result.groupby(['year', 'week']).apply(get_prediciton_score, metric=met)
        best_result_month = best_result.groupby(['year', 'month']).apply(get_prediciton_score, metric=met)
        best_result_year = best_result.groupby(['year']).apply(get_prediciton_score, metric=met)
        
        best_row=best_result_day.sort_values(ascending=False).index[0]
        best_day_data = best_result[(best_result['year'] == best_row[0])  & (best_result['week'] == best_row[1]) & (best_result['day'] == best_row[2])]

        best_row=best_result_week.sort_values(ascending=False).index[0]
        best_week_data = best_result[(best_result['year'] == best_row[0]) & (best_result['week'] == best_row[1])]

        best_row=best_result_month.sort_values(ascending=False).index[0]
        best_month_data = best_result[(best_result['year'] == best_row[0]) & (best_result['month'] == best_row[1]) ]

        best_row=best_result_year.sort_values(ascending=False).index[0]
        best_year_data = best_result[(best_result['year'] == best_row)]

        worst_row=best_result_day.sort_values(ascending=True).index[0]
        worst_day_data = best_result[(best_result['year'] == worst_row[0])  & (best_result['week'] == worst_row[1]) & (best_result['day'] == worst_row[2])]
        worst_row=best_result_week.sort_values(ascending=True).index[0]
        worst_week_data = best_result[(best_result['year'] == worst_row[0]) & (best_result['week'] == worst_row[1])]
        worst_row=best_result_month.sort_values(ascending=True).index[0]
        worst_month_data = best_result[(best_result['year'] == worst_row[0]) & (best_result['month'] == worst_row[1]) ]
        worst_row=best_result_year.sort_values(ascending=True).index[0]
        worst_year_data = best_result[(best_result['year'] == worst_row)]

    
        time_frame_images(worst_year_data, worst_month_data, worst_week_data, worst_day_data, name=f"Worst {name}_{met}", folder_figures=folder_to_save)
        time_frame_images(best_year_data, best_month_data, best_week_data, best_day_data, name=f"Best {name}_{met}",folder_figures=folder_to_save)
    pp = prediction_score(best_result["test"], best_result["prediction"], best_result["benchmark"], "name")


def write_time_frames(experiment, **kwargs):
    exp_results = experiment.validate_experiment()
    unique_name = exp_results["name"].unique()
    benchmark_score = None
    benchmark_data=None
    if benchmark_data is None:
        benchmark_data = os.path.join(
            experiment.data_manager.work_folder,
            "benchmark",
            experiment.target_variable,
            "benchmark.json",
        )
        benchmark_score = load_json_dict(benchmark_data)
        benchmark_data = np.load(benchmark_data.replace("json", "npz"))

    folder_figures = kwargs.pop(
        "folder_figures",
        experiment.work_folder.replace("/experiment/", "/reports/"),
    )
    figure_name = kwargs.pop(
        "figure_name", f"time_frame_validations_{experiment.target_variable}.png"
    )
    validation_target = experiment.validation_target

    for name in unique_name:
        best_case, best_result = experiment.result_validation_fn(
            exp_results[exp_results["name"]==name], validation_target, **kwargs
        )
        epocha=best_case["epoch"].item()
        best_result_path = os.path.join(experiment.model_handler.work_folder, name,"freq_predictions", f"{epocha}.npz")
        time_frame_validations(best_result_path, name, validation_target, folder_to_save=folder_figures)
