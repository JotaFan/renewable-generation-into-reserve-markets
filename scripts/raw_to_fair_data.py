import os
from pathlib import Path
from typing import Union

import pandas as pd

RAW_DATA_FOLDER = Path("data/raw_data")
TREATED_DATA_FOLDER = Path("data")
beta_study_raw = RAW_DATA_FOLDER / "Secondary.xlsx"

consuption_study_raw = RAW_DATA_FOLDER / "DynamicSecondary.xlsx"


beta_metadata = [
    {"name": "DATA", "units": "datetime"},
    {"name": "HORA", "units": "int"},
    {"name": "BANDA_SUBIR", "units": "MW"},
    {"name": "BANDA_DESCER", "units": "MW"},
    {"name": "Consumo real", "units": "MWh"},
    {"name": "Consumo Máximo ENTSO-E", "units": "MWh"},
]

consuption_metadata = [
    {"name": "Energia Down", "units": "MW"},
    {"name": "Energia Up", "units": "MW"},
    {"name": "Day-ahead Forecast Wind", "units": "MWh"},
    {"name": "DA Forecast PV", "units": "MWh"},
    {"name": "DA Forecast Consumo", "units": "MWh"},
    {"name": "DAToda a Geração", "units": "MWh"},
    {"name": "DA Tie Lines Balance", "units": "MW"},
    {"name": "DA Traded Wind", "units": "MWh"},
    {"name": "DA Traded PV", "units": "MWh"},
]


def treat_data(
    path_raw_data: Union[Path, str],
    variables_descriptions: list,
    path_treated_data: Union[Path, str] = None,
):
    raw_data = pd.read_excel(path_raw_data)
    extension = ".xlsx"

    # assert set(raw_data.columns) == set(
    #     variables_descriptions[0]
    # ), "Columns on data and descriptions does not match"

    metadata_table = pd.DataFrame(variables_descriptions)

    if not path_treated_data:
        filename = os.path.basename(path_raw_data)
        # filename = variables_descriptions.get("name", filename)
        filename = filename.split(extension)[0]
        path_treated_data = TREATED_DATA_FOLDER / f"{filename}.csv"
    metada_filename = f"{filename}_metadata.csv"
    path_treated_data_metadata = TREATED_DATA_FOLDER / metada_filename

    raw_data.to_csv(path_treated_data)
    metadata_table.to_csv(path_treated_data_metadata)

treat_data(beta_study_raw, beta_metadata)
treat_data(consuption_study_raw, consuption_metadata)