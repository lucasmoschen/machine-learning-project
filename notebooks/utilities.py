#!/usr/bin/python

import json
import os
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as r2
import numpy as np

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