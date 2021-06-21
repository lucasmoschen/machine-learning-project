#!/usr/bin/python

import json
import os
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as r2
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

class Utilities: 

    def __init__(self): 
        pass 

    def save_metrics(self, gas, station, model_name, hyperparameters, 
                     r2_train, r2_test, mae_train, mae_test, rmse_train, rmse_test): 

        if not os.path.exists('../data/models.json'): 
            with open('../data/models.json', 'w') as f:
                json.dump({}, f)    

        with open('../data/models.json', 'r', encoding='utf-8') as f:
            models = json.load(f)

        i = len(models.keys())
        models[i] = {
            "gas": gas,
            "station": station, 
            "model_name": model_name, 
            "hyperparameters": hyperparameters, 
            "r2_train": r2_train, 
            "r2_test": r2_test, 
            "mae_train": mae_train, 
            "mae_test": mae_test, 
            "rmse_train": rmse_train, 
            "rmse_test": rmse_test
        }

        with open('../data/models.json', 'w', encoding='utf-8') as f: 
            json.dump(models, f)

        return

    def print_results(self, r2_train, r2_test, mae_train, mae_test, rmse_train, rmse_test): 

        print("R2 train set: {}".format(r2_train))
        print("MAE train set: {}".format(mae_train))
        print("RMSE train set: {}".format(rmse_train))

        print("R2 test set: {}".format(r2_test))
        print("MAE test set: {}".format(mae_test))
        print("RMSE test set: {}".format(rmse_test))


    def forward_selection(self, data, target, regr, k_fold, threshold):

        kf = KFold(n_splits=k_fold)
        features = data.columns.tolist()
        best_features = []

        r2_general = 0.0

        while len(features) > 0:

            max_r2 = -np.inf
            for feat in features: 

                X = data[best_features + [feat]]
                r2_error = 0
                for n, (train_index, validation_index) in enumerate(kf.split(data)):

                    x_train, x_test = X.iloc[train_index], X.iloc[validation_index]
                    y_train, y_test = target.iloc[train_index], target.iloc[validation_index]

                    regr.fit(x_train, y_train)
                    y_pred = regr.predict(x_test)
                    r2_error = (n*r2_error + r2(y_test, y_pred))/(n+1)
                
                if r2_error > max_r2: 
                    max_r2 = r2_error
                    chosen_feature = feat
            
            best_features.append(chosen_feature)
            features.remove(chosen_feature)

            text = "The maximum R2 until now is {} with {} featur{}."
            additional = 'es' if len(best_features) > 1 else 'e'
            print(text.format(max_r2, len(best_features), additional), end = "\r")

            if max_r2 - r2_general < threshold:
                break
            else: 
                r2_general = max_r2
        
        return best_features
    
    def linear_regression_em_preparation(self, location): 
        """
        All preparations with no missing value imputation.
        """
        
        air_data = pd.read_csv(location + "RiodeJaneiro_MonitorAr_hourly_p1.csv", index_col = 0)
        air_data.Data = pd.to_datetime(air_data.Data) 

        air_data = air_data[air_data.year < 2020]

        air_data["weekend"] = (air_data.Data.dt.weekday >= 5).astype(int)
        air_data["season"] = (air_data.month - 1)// 3
        air_data.season += (air_data.month == 3)&(air_data.day>=20)
        air_data.season += (air_data.month == 6)&(air_data.day>=21)
        air_data.season += (air_data.month == 9)&(air_data.day>=23)
        air_data.season += (air_data.month == 12)&(air_data.day>=21)
        air_data.season = air_data.season%4 

        air_data["hour_sin"] = np.sin(air_data.hour*(2*np.pi/24))
        air_data["hour_cos"] = np.cos(air_data.hour*(2*np.pi/24))

        var_continuous = ['Chuva', 'Pres', 'RS', 'Temp', 'UR', 'Dir_Vento', 'Vel_Vento', 'CO', 'O3', 'PM10']
        pt = PowerTransformer(method = 'yeo-johnson', standardize=True).fit(air_data[var_continuous])
        transform_air_data = pt.transform(air_data[var_continuous])

        air_data[var_continuous] = transform_air_data

        air_data.sort_values(["year", "month", "day", "hour"], inplace = True)

        datas = air_data.Data.unique()
        map_train_test = {date: date <= datas[int(0.7*datas.shape[0])+1] for date in datas}

        air_data["train"] = air_data.Data.map(map_train_test)

        for gas in ["CO", "O3", "PM10"]:
            for lag in ["1","2","24"]: 
                air_data[gas+"_lag"+lag] = air_data[gas]
            air_data[gas+"_MA24"] = air_data[gas]

        for station in range(1,9):
            for gas in ["CO", "O3", "PM10"]: 
                for lag in [1,2,24]:
                    df =  air_data.loc[air_data.CodNum == station, gas+"_lag"+str(lag)].shift(lag)
                    air_data.loc[air_data.CodNum == station, gas+"_lag"+str(lag)] = df
                df = air_data.loc[air_data.CodNum == station, gas+"_MA24"].rolling(window = 24).mean()
                air_data.loc[air_data.CodNum == station, gas+"_MA24"] = df

        air_data.drop(columns=["Data", "hour"], inplace=True)
        
        df_train = air_data[air_data.train].drop(columns='train')
        x_train = df_train.drop(columns=["O3", 'CO', 'PM10', 'Lat', 'Lon'])
        x_train_SP = x_train[x_train.CodNum == 8].drop(columns="CodNum")
        x_train_SP['O3'] = df_train[df_train.CodNum==8].O3
        
        return x_train_SP