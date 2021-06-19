#!/usr/bin/python

import json
import os

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


        