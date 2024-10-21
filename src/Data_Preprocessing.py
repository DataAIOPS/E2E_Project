import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import argparse
import mlflow

with open("./../config/config.yaml","r") as file:
    config =  yaml.safe_load(file)

test_size = config['preprocess']['test_size']
Target = config['preprocess']['Target']
processed_data_path = config['data_paths']['processed_data_path']
clean_data_path = config['data_paths']['clean_data_path']

def processed_data(cleaned_data_path, processed_data_path, Target):
    print("################DATA PREPROCESSING STARTED#####################")
    cleaned_data_file = os.listdir(cleaned_data_path)[0]
    cleaned_data = os.path.join(cleaned_data_path,cleaned_data_file)
    df = pd.read_csv(cleaned_data)
    X = df.drop(columns=[Target])
    Y = df[[Target]]

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=test_size)

    mlflow.log_param("test_size",test_size)

    X_train.to_csv(os.path.join(processed_data_path,"X_train.csv"))
    X_test.to_csv(os.path.join(processed_data_path,"X_test.csv"))
    y_train.to_csv(os.path.join(processed_data_path,"y_train.csv"))
    y_test.to_csv(os.path.join(processed_data_path,"y_test.csv"))
    print("################DATA PREPROCESSING FINISHED#####################")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_data_path",help="provide raw data path", default=clean_data_path)
    parser.add_argument("--processed_data_path",help="provide cleaned data path", default=processed_data_path)
    parser.add_argument("--Target",help="provide cleaned data path", default=Target)
    args = parser.parse_args()
    processed_data(args.cleaned_data_path,args.processed_data_path,args.Target)
    