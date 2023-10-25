import pandas as pd
import numpy as np
import os

path ="/home/joao/Documentos/repos/renewable-generation-into-reserve-markets/models_validation/losses_experiment/StackedCNNwl/freq_saves/StackedCNNwl131epc_test.npz"

experiment_name = os.path.dirname(os.path.dirname(path))
experiment_name_epoc = 131
experiment_name = f"test_{experiment_name}.csv"

np_saved = np.load(path)
df = pd.DataFrame()
df["teste"] = np_saved.get("test").ravel()
df["prediction"] = np_saved.get("prediction").ravel()
df.to_csv(experiment_name)
