from pyesios import ESIOS
import datetime
import pandas as pd
import json

# The token is unique: You should ask for yours to:
#   Consultas Sios <consultasios@ree.es>
token = "631a6970d32bd6eab93398b89246b642b859e541894deef5b7605c1f6f4c9d5c"



esios = ESIOS(token)

indicators_ = [
    632,
    633,
    680,
    681,
    1777,
    1779,
    1775,
    10258,
    14,
    10073,
    10186,
]

names = esios.get_names(indicators_)

indicators_df = {
    "indicators" : indicators_,
    "names":names,
}

print("Getting the indicators:", names)



# Trying to get older dates, visualy verified no earlier than 2012
#   exists on targe data
start_ = "01-01-2012T00"
end_ = "01-01-2022T00"
start_date = datetime.datetime.strptime(start_, "%d-%m-%YT%H")
end_date = datetime.datetime.strptime(end_, "%d-%m-%YT%H")

# The dates are distribuited per max two years.
# This is because the API breaks when asking more than two years at a time.
list_dates = [
    (f + datetime.timedelta(days=1)).strftime("%d-%m-%YT%H")
    for f in pd.date_range(start_date, end_date, freq="Y")
]




lll = []
for i in range(len(list_dates) - 1):
    sss = list_dates[i]
    eee = list_dates[i + 1]
    try:
        dfmul = esios.get_multiple_series(indicators_, sss, eee)

        ff_df = dfmul[0][0]
        for i in range(len(dfmul[0])):
            df = dfmul[0][i]
            if type(df) == type(None):
                continue
            cols_to_use = list(df.columns.difference(ff_df.columns))
            if len(cols_to_use) == 0:
                continue
            cols_to_use.append([f for f in df.columns if "datetime" in f][0])
            ff_df = ff_df.merge(
                df[cols_to_use], on="datetime", how="left"
            ).reset_index(drop=True)
        lll.append(ff_df)
    except Exception as e:
        print("wriong", e)

all_data = pd.concat(lll).drop_duplicates().reset_index(drop=True)

# Translate the names to english
merge_dict = {
    "DownwardUsedSecondaryReserveEnergy": "Energía utilizada de Regulación Secundaria bajar",
    "UpwardUsedSecondaryReserveEnergy": "Energía utilizada de Regulación Secundaria subir",
    "BaseDailyOperatingSchedulePBFSolarPV": "Generación programada PBF Solar fotovoltaica",
    "BaseDailyOperatingSchedulePBFWind": "Generación programada PBF Eólica",
    "BaseDailyOperatingShedulePBFTotalBalanceInterconnections": "Saldo total interconexiones programa PBF",
    "DemandD+1DailyForecast": "Previsión diaria D+1 demanda",
    "PhotovoltaicD+1DailyForecast": "Previsión diaria D+1 fotovoltaica",
    "SecondaryReserveAllocationADownward": "Asignación Banda de regulación secundaria a bajar",
    "SecondaryReserveAllocationAUpward": "Asignación Banda de regulación secundaria a subir",
    "TotalBaseDailyOperatingSchedulePBFGeneration": "Generación programada PBF total",
    "WindD+1DailyForecast": "Previsión diaria D+1 eólica",
    "datetime": "datetime",
}

rename_dict = {y: x for x, y in merge_dict.items()}
indicators_df["names"] = [rename_dict[f] for f in indicators_df["names"]]
indicators_json = []
for name, indi in zip(indicators_df["names"], indicators_df["indicators"]):
    indicators_json.append({
        "name":name,
        "indicator":indi,
        "description":"",
        "units":"",
        "url":f"https://www.esios.ree.es/en/analysis/{indi}",
        "url_semantic":"",
    })
with open("data/indicators_metadata.json", "w") as mfile:
    json.dump(indicators_json, mfile)



# save to csv for later metadate prep.abs
pd.DataFrame(indicators_df).to_csv("data/indicators_metadata.csv")

all_data_rename = all_data.rename(columns=rename_dict)

all_data_rename = all_data_rename.drop_duplicates("datetime").drop(["tz_time", "geo_id", "geo_name"], axis=1).reset_index(drop=True)

all_data_rename.to_csv("data/dados_2014-2022.csv")