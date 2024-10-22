import mlflow
from mlflow.tracking import MlflowClient
import os

def model_fecthing(best_model_path,experiment_name='Default'):
    clinet = MlflowClient()

    experiment = clinet.get_experiment_by_name(experiment_name)
    print(f"Experiment={experiment}")
    if experiment is None:
        raise ValueError(f"Experiment:{experiment_name} not found")
    
    experiment_id = experiment.experiment_id
    print(experiment_id)

    runs = clinet.search_runs(experiment_ids=[experiment_id],filter_string="tags.mlflow.runName = 'Model_Evaluation'")
    print(runs)
    best_MAE = float("inf")
    best_r2_score=float("-inf")
    best_run=0
    for run in runs:
        metrics =  run.data.metrics
        MAE_test = metrics.get('MAE_test',float("inf"))
        r2_score_test = metrics.get('r2_score_test',float("inf"))

        if MAE_test < best_MAE and r2_score_test> best_r2_score:
            best_MAE = MAE_test
            best_r2_score = r2_score_test
            best_run = run

    if best_run:
        best_run_id = best_run.info.run_id
        print(f"Best run ID: {best_run_id} with MAE: {best_MAE} and R²: {best_r2_score}")

        model_uri = f"runs:/{best_run_id}/linear_reg"
        local_model_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=best_model_path)
                
        print(f"Model downloaded to: {local_model_path}")


best_model_path="./artifacts/model/best_model"
experiment_name = "New"
model_fecthing(best_model_path,experiment_name)    

    

    

