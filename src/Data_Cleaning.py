import pandas as pd
import argparse
import os
import yaml

with open("./config/config.yaml","r") as file:
    config =  yaml.safe_load(file)

raw_data_path = config['data_paths']['raw_data_path']
clean_data_path = config['data_paths']['clean_data_path']


def data_cleaning(raw_data_path, clean_data_path, raw_data_file):
    raw_data_file = os.listdir(raw_data_path)[0]
    print("################DATA CLEANING STARTED##################")
    raw_data = os.path.join(raw_data_path,raw_data_file)
    df = pd.read_csv(raw_data)
    
    '''
    You can write some data cleaning steps here and your cleaned data will be exported

    '''

    clean_data_file = os.path.join(clean_data_path,raw_data_file)
    df.to_csv(clean_data_file,index=False)
    print("################DATA CLEANING FINISHED##################")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path",help="provide raw data path", default=raw_data_path)
    parser.add_argument("--clean_data_path",help="provide cleaned data path", default=clean_data_path)
    args = parser.parse_args()
    data_cleaning(args.raw_data_path,args.clean_data_path)



