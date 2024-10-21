import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import pickle
import yaml
import argparse

with open("./config/config.yaml","r") as file:
    config =  yaml.safe_load(file)

processed_data_path = config['data_paths']['processed_data_path']
model_path = config['data_paths']['model_path']
fit_intercept = config['model']['fit_intercept']


def model_building(processed_data,model_path):
    print("################MODEL BUILDING STARTED#####################")

    x_train_path = os.path.join(processed_data,"X_train.csv")
    y_train_path = os.path.join(processed_data,"y_train.csv")

    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)

    print("[INFO] Model building is started")
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(x_train,y_train)

    model_path_file = os.path.join(model_path,"linear_model.pkl")
    pickle.dump(model,open(model_path_file,"wb"))
    

    print(f"[INFO] model is exporeted")
    print("################MODEL BUILDING FINISHED#####################")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_path",help="provide raw data path", default=processed_data_path)
    parser.add_argument("--model_path",help="provide cleaned data path", default=model_path)
    args = parser.parse_args()
    model_building(args.processed_data_path,args.model_path)