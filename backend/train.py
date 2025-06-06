# =========================================
# H2O AutoML Training with MLflow Tracking
# Author: Kenneth Leung (adapté par Folly Ayeboua)
# =========================================

import os
import argparse
import h2o
from h2o.automl import H2OAutoML, get_leaderboard

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient

import json
import pandas as pd  # pour manipuler le leaderboard en DataFrame
import matplotlib.pyplot as plt

# -----------------------------------------
# 1) On configure MLflow pour pointer vers le serveur MLflow
#    (service "mlflow" défini dans docker-compose.yml).
# -----------------------------------------
mlflow.set_tracking_uri("http://mlflow:5001")

def train(experiment_name: str, target: str, models: int):

    h2o.init()

    chemin_train = 'data/processed/train.csv'
    if not os.path.exists(chemin_train):
        raise FileNotFoundError(f"Fichier d'entraînement introuvable : {chemin_train}")

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

        # Artefacts : données
        mlflow.log_artifact(chemin_train, artifact_path="input_data")
        mlflow.log_artifact(chemin_col_types, artifact_path="input_data")

        # Artefact : config JSON
        config = {
            "experiment": experiment_name,
            "target": target,
            "max_models": models,
            "predictors": predictors
        }
        os.makedirs("/tmp/automl_config", exist_ok=True)
        config_path = "/tmp/automl_config/config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(config_path, artifact_path="config")

        # H2O AutoML
        aml = H2OAutoML(
            max_models=models,
            seed=42,
            balance_classes=True,
            sort_metric='logloss',
            verbosity='info',
            exclude_algos=['GLM', 'DRF']
        )
        aml.train(x=predictors, y=target, training_frame=main_frame)

        perf = aml.leader.model_performance()

        # Métriques
        mlflow.log_metric("logloss", perf.logloss())
        mlflow.log_metric("AUC", perf.auc())
        mlflow.log_metric("accuracy", perf.accuracy()[0][1])
        mlflow.log_metric("mean_per_class_error", perf.mean_per_class_error())
        mlflow.log_metric("rmse", perf.rmse())
        mlflow.log_metric("mse", perf.mse())

        # Modèle H2O via MLflow
        mlflow.h2o.log_model(aml.leader, artifact_path="model")

        # Modèle H2O sauvegardé manuellement
        saved_model_path = h2o.save_model(model=aml.leader, path="/tmp", force=True)
        mlflow.log_artifact(saved_model_path, artifact_path="h2o_model_bin")

        # Leaderboard
        leaderboard = get_leaderboard(aml, extra_columns="ALL").as_data_frame()
        leaderboard_path = f"/tmp/leaderboard_{run_id}.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        mlflow.log_artifact(leaderboard_path, artifact_path="leaderboard")

        # Graphique : importance des variables
        varimp_path = f"/tmp/varimp_{run_id}.png"
        plt.figure(figsize=(10, 6))
        h2o.plot_varimp(aml.leader, num_of_features=10, figsize=(10, 6), server=False)
        plt.tight_layout()
        plt.savefig(varimp_path)
        mlflow.log_artifact(varimp_path, artifact_path="figures")

        # Graphique : courbe ROC
        roc_path = f"/tmp/roc_curve_{run_id}.png"
        plt.figure()
        perf.plot(type="roc", server=False)
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path, artifact_path="figures")

        print(f"[✓] Entraînement terminé. Run ID : {run_id}")
        print("[✓] Modèle, métriques et artefacts loggés dans MLflow.")

        run_id = run.info.run_id  # ID du run pour information complémentaire
        print("→ Run démarré :", run.info.run_id)


        # 5.1) Enregistrement des données d'entrée comme artefacts
        #     (le serveur MLflow s'occupera de les copier dans ARTIFACT_ROOT)
        mlflow.log_artifact(chemin_train, artifact_path="input_data")
        mlflow.log_artifact(chemin_col_types, artifact_path="input_data")

        # 5.2) Configuration et exécution de H2O AutoML
        aml = H2OAutoML(
            max_models=models,
            seed=42,
            balance_classes=True,
            sort_metric='logloss',
            verbosity='info',
            exclude_algos=['GLM', 'DRF']
        )
        aml.train(x=predictors, y=target, training_frame=main_frame)

        # 5.3) Log des métriques principales
        #     (on peut en logguer autant que nécessaire)
        mlflow.log_metric("log_loss", aml.leader.logloss())
        mlflow.log_metric("AUC", aml.leader.auc())

        # 5.4) Log du modèle H2O dans MLflow
        #     mlflow.h2o.log_model transfère le binaire du modèle dans l'artefact
        mlflow.h2o.log_model(aml.leader, artifact_path="model")

        # 5.5) Construction du leaderboard en DataFrame et enregistrement
        lb = get_leaderboard(aml, extra_columns='ALL')
        # Convertit en pandas DataFrame
        df_leaderboard = lb.as_data_frame()

        # On sauvegarde temporairement en local (dans /tmp)
        temp_dir = "/tmp/mlflow_leaderboard"
        os.makedirs(temp_dir, exist_ok=True)
        leaderboard_csv = os.path.join(temp_dir, f"leaderboard_{run_id}.csv")
        df_leaderboard.to_csv(leaderboard_csv, index=False)

        # Puis on loggue ce CSV comme artefact “leaderboard”
        mlflow.log_artifact(leaderboard_csv, artifact_path="leaderboard")

        print(f"[✓] Modèle et leaderboard loggés dans MLflow (run_id={run_id})")

def parse_args():
    parser = argparse.ArgumentParser(
        description="H2O AutoML Training and MLflow Tracking"
    )

    parser.add_argument(
        '--name', '--experiment_name',
        metavar='',
        default='automl-insurance',
        help='Nom de l’expérience MLflow (par défaut: automl-insurance)',
        type=str
    )

    parser.add_argument(
        '--target', '--t',
        metavar='',
        required=True,
        help='Nom de la colonne cible (y)',
        type=str
    )

    parser.add_argument(
        '--models', '--m',
        metavar='',
        default=10,
        help='Nombre de modèles AutoML à entraîner (par défaut: 10)',
        type=int
    )

    return parser.parse_args()

def main():
    # 1) Analyse des arguments
    args = parse_args()
    experiment_name = args.name
    target = args.target
    models = args.models

    # 2) Initialisation du cluster H2O (il reste ouvert tout au long du script)
    h2o.init()

    # 3) Affichage de quelques infos sur l’expérience MLflow
    client = MlflowClient()
    try:
        exp_id = mlflow.create_experiment(experiment_name)
        experiment = client.get_experiment(exp_id)
    except Exception:
        experiment = client.get_experiment_by_name(experiment_name)

    print(f"Nom de l'expérience : {experiment_name}")
    print(f"ID: {experiment.experiment_id}")
    print(f"URI des artefacts : {experiment.artifact_location}")
    print(f"Statut: {experiment.lifecycle_stage}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    # 4) Chargement et préparation des données (identique à la fonction train)
    main_frame = h2o.import_file(path='data/processed/train.csv')
    with open('data/processed/train_col_types.json', 'w') as fp:
        json.dump(main_frame.types, fp)
    predictors = [n for n in main_frame.col_names if n != target]
    main_frame[target] = main_frame[target].asfactor()

    # 5) Démarrer un run MLflow pour ce train
    with mlflow.start_run():
        aml = H2OAutoML(
            max_models=models,
            seed=42,
            balance_classes=True,
            sort_metric='logloss',
            verbosity='info',
            exclude_algos=['GLM', 'DRF']
        )
        aml.train(x=predictors, y=target, training_frame=main_frame)

        # Log metrics
        mlflow.log_metric("log_loss", aml.leader.logloss())
        mlflow.log_metric("AUC", aml.leader.auc())

        # Log model
        mlflow.h2o.log_model(aml.leader, artifact_path="model")

        # Leaderboard
        lb = get_leaderboard(aml, extra_columns='ALL')
        df_leaderboard = lb.as_data_frame()
        temp_dir = "/tmp/mlflow_leaderboard"
        os.makedirs(temp_dir, exist_ok=True)
        leaderboard_csv = os.path.join(temp_dir, "leaderboard.csv")
        df_leaderboard.to_csv(leaderboard_csv, index=False)
        mlflow.log_artifact(leaderboard_csv, artifact_path="leaderboard")

        print("Entraînement AutoML terminé. Modèle et leaderboard loggés.")

if __name__ == "__main__":
    main()
