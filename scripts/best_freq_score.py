import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

COLUMN_TO_SORT_BY = "optimal percentage"
ascending_to_sort = False

COLUMN_TO_SORT_BY = "alloc missing"
if COLUMN_TO_SORT_BY =="alloc missing":
    ascending_to_sort = True
#scripts_path = os.path.dirname(os.path.abspath(__file__))


#sys.path.insert(0, scripts_path) # Add the script's directory to the system path
path_to_score_folder = "/home/joao/Documentos/repos/renewable-generation-into-reserve-markets/models_validation/linear_models_epocs/"
path_to_score_folder = "/home/joao/Documentos/repos/renewable-generation-into-reserve-markets/models_validation/losses_experiment/"

list_of_dirs = [os.path.join(path_to_score_folder,f) for f in os.listdir(path_to_score_folder) if os.path.isdir(os.path.join(path_to_score_folder,f))]
list_of_dirs = [f for f in list_of_dirs if "cache" not in f]

list_of_dirs1 = [f for f in list_of_dirs if "UNET" in f]
list_of_dirs2 = [f for f in list_of_dirs if "StackedCNN" in f]
list_of_dirs = list_of_dirs1+list_of_dirs2

all_scores = pd.DataFrame()
for exp in list_of_dirs:
    scores_path = os.path.join(exp, "freq_saves","experiment_results_complete.csv")
    scores_pd = pd.read_csv(scores_path)
    scores_pd["name"] = os.path.basename(exp)
    all_scores = pd.concat([all_scores, scores_pd], ignore_index=True)

# Sort the DataFrame by "optimal percentage" column in descending order
all_scores.sort_values(by=COLUMN_TO_SORT_BY, ascending=False, inplace=True)

# Get the best score for each unique value in the "name" column
best_scores = all_scores.loc[all_scores.groupby("name").idxmax()[COLUMN_TO_SORT_BY]].sort_values(by=COLUMN_TO_SORT_BY, ascending=ascending_to_sort)

path_schema_tex = os.path.join(path_to_score_folder, "experiment_results_best_of_each.tex")

best_scores.to_latex(path_schema_tex, escape=False,index=False, float_format="%.2f")


# Get the 2nd and 3rd best scores for each unique value in the "name" column
second_third_best_scores = all_scores.groupby("name").apply(lambda x: x.nsmallest(3, COLUMN_TO_SORT_BY))

second_third_best_scores.sort_values(by=COLUMN_TO_SORT_BY, ascending=ascending_to_sort, inplace=True)

path_schema_tex = os.path.join(path_to_score_folder, "experiment_results_best3.tex")

second_third_best_scores.to_latex(path_schema_tex, escape=False,index=False, float_format="%.2f")


benchmark_alloc_missing=146915.2
benchmark_alloc_surplus=30178735.2

all_scores['alloc missing smaller'] = all_scores['alloc missing'] <= benchmark_alloc_missing
all_scores['alloc surplus smaller'] = all_scores['alloc surplus'] <= benchmark_alloc_surplus

all_scores["bscore m"] = ((benchmark_alloc_missing - all_scores['alloc missing'])/benchmark_alloc_missing)*100
all_scores["bscore s"] = ((benchmark_alloc_surplus - all_scores['alloc surplus'])/benchmark_alloc_surplus)*100
all_scores["bscore"] = all_scores["bscore s"] + all_scores["bscore m"]

no_missin_scores = all_scores[all_scores["bscore m"]>=0]
no_missin_scores = no_missin_scores[no_missin_scores["bscore s"]>=0]
no_missin_scores = no_missin_scores.dropna().sort_values(by="bscore", ascending=False)

path_schema_tex = os.path.join(path_to_score_folder, "experiment_results_best_under_benchmark.tex")

no_missin_scores.head(10).to_latex(path_schema_tex, escape=False,index=False, float_format="%.2f")


no_missin_scores = no_missin_scores.groupby("name").apply(lambda x: x.nlargest(3, "bscore"))

no_missin_scores.sort_values(by="bscore", ascending=False, inplace=True)

path_schema_tex = os.path.join(path_to_score_folder, "experiment_results_best3_under_benchmark.tex")

no_missin_scores.to_latex(path_schema_tex, escape=False,index=False, float_format="%.2f")
