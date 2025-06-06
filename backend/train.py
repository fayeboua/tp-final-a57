import os
import argparse
import h2o
from h2o.automl import H2OAutoML, get_leaderboard
import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
import json
import pandas as pd

mlflow.set_tracking_uri("http://mlflow:5001")

def train(experiment_name: str, target: str, models: int):
    h2o.init()

    chemin_train = 'data/processed/train.csv'
    if not os.path.exists(chemin_train):
        raise FileNotFoundError(f"Fichier introuvable : {chemin_train}")
    main_frame = h2o.import_file(path=chemin_train)

    chemin_col_types = 'data/processed/train_col_types.json'
    with open(chemin_col_types, 'w') as fp:
        json.dump(main_frame.types, fp)

    predictors = [col for col in main_frame.col_names if col != target]
    main_frame[target] = main_frame[target].asfactor()

    client = MlflowClient()
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)

    print("→ Lancement du run MLflow pour l’expérience:", experiment_name)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print("→ Run ID :", run_id)

        mlflow.log_artifact(chemin_train, artifact_path="input_data")
        mlflow.log_artifact(chemin_col_types, artifact_path="input_data")

        aml = H2OAutoML(
            max_models=models,
            seed=42,
            balance_classes=True,
            sort_metric='logloss',
            verbosity='info',
            exclude_algos=['GLM', 'DRF']
        )
        aml.train(x=predictors, y=target, training_frame=main_frame)

        leader = aml.leader
        perf = leader.model_performance()

        # Enregistrement des métriques avec vérification de type
        mlflow.log_metric("logloss", float(perf.logloss()))
        mlflow.log_metric("AUC", float(perf.auc()))
        mlflow.log_metric("mean_per_class_error", float(perf.mean_per_class_error()))
        mlflow.log_metric("rmse", float(perf.rmse()))
        mlflow.log_metric("mse", float(perf.mse()))

        # Accuracy nécessite extraction depuis la liste retournée
        accuracy_list = perf.accuracy()
        if isinstance(accuracy_list, list) and len(accuracy_list) > 0:
            mlflow.log_metric("accuracy", float(accuracy_list[0][1]))

        mlflow.h2o.log_model(leader, artifact_path="model")

        lb = get_leaderboard(aml, extra_columns='ALL')
        df_leaderboard = lb.as_data_frame()
        temp_dir = "/tmp/mlflow_leaderboard"
        os.makedirs(temp_dir, exist_ok=True)
        leaderboard_csv = os.path.join(temp_dir, f"leaderboard_{run_id}.csv")
        df_leaderboard.to_csv(leaderboard_csv, index=False)
        mlflow.log_artifact(leaderboard_csv, artifact_path="leaderboard")

        print(f"[✓] Modèle et leaderboard loggés dans MLflow (run_id={run_id})")

def parse_args():
    parser = argparse.ArgumentParser(description="H2O AutoML avec MLflow")
    parser.add_argument('--name', default='automl-insurance', help='Nom de l’expérience MLflow')
    parser.add_argument('--target', required=True, help='Nom de la colonne cible')
    parser.add_argument('--models', default=10, type=int, help='Nombre de modèles à entraîner')
    return parser.parse_args()

def main():
    args = parse_args()
    train(experiment_name=args.name, target=args.target, models=args.models)

if __name__ == "__main__":
    main()
