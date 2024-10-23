import argparse
import mlflow
import os
import time
import yaml

with open("./config/config.yaml","r") as file:
    config =  yaml.safe_load(file)

with open("./config/secrets.yaml","r") as file:
    secrets =  yaml.safe_load(file)

os.environ['MLFLOW_TRACKING_URI'] = secrets['mlflow']['MLFLOW_TRACKING_URI']
os.environ['MLFLOW_TRACKING_USERNAME'] = secrets['mlflow']['MLFLOW_TRACKING_USERNAME']
os.environ['MLFLOW_TRACKING_PASSWORD'] = secrets['mlflow']['MLFLOW_TRACKING_PASSWORD']

run_name = config['mlflow']['run_name']

mlflow.set_experiment("test")

def main(run_name,pipeline):
    print("[INFO] MLOps Pipeline Triggerd")
    with mlflow.start_run(run_name=run_name):
        if pipeline == "training":
            mlflow.run("./src",entry_point="Data_Cleaning.py",env_manager="local",run_name="Data_Cleaning")
            mlflow.run("./src",entry_point="Data_Preprocessing.py",env_manager="local",run_name="Data_Preprocessing")
            mlflow.run("./src",entry_point="Model_Building.py",env_manager="local",run_name="Model_Building")
            mlflow.run("./src",entry_point="Model_Evaluation.py",env_manager="local",run_name="Model_Evaluation")
        elif pipeline == "fetching":
            mlflow.run("./src",entry_point="Model_Fetching.py",env_manager="local",run_name="Model_Fetching")
        else:
            print("Please select pipeline from training or fetching")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",type=str, default=run_name)
    parser.add_argument("--pipeline",type=str)
    args = parser.parse_args()
    main(args.run_name,args.pipeline)
