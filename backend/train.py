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

# -----------------------------------------
# 1) On configure MLflow pour pointer vers le serveur MLflow
#    (service "mlflow" défini dans docker-compose.yml).
# -----------------------------------------
mlflow.set_tracking_uri("http://mlflow:5001")

def train(experiment_name: str, target: str, models: int):
    """
    Lance un run H2O AutoML et enregistre le modèle + le leaderboard
    dans MLflow via le serveur défini sur http://mlflow:5001.
    """
    # -----------------------------------------
    # 2) Initialisation du cluster H2O
    # -----------------------------------------
    h2o.init()

    # -----------------------------------------
    # 3) Chargement des données d'entraînement
    # -----------------------------------------
    chemin_train = 'data/processed/train.csv'
    if not os.path.exists(chemin_train):
        raise FileNotFoundError(f"Fichier d'entraînement introuvable : {chemin_train}")

    # On importe en tant que H2OFrame
    main_frame = h2o.import_file(path=chemin_train)

    # Sauvegarder localement les types de colonnes (pour le servir plus tard si nécessaire)
    chemin_col_types = 'data/processed/train_col_types.json'
    with open(chemin_col_types, 'w') as fp:
        json.dump(main_frame.types, fp)

    # On définit la liste des prédicteurs
    predictors = [col for col in main_frame.col_names if col != target]
    # On convertit la colonne cible en facteur pour classification
    main_frame[target] = main_frame[target].asfactor()

    # -----------------------------------------
    # 4) Initialisation de l'expérience MLflow
    # -----------------------------------------
    client = MlflowClient()  # client qui va interroger le serveur MLflow
    try:
        exps = client.search_experiments()  # ou client.list_experiments() selon votre version MLflow
        print("Liste des expériences trouvées :", [e.name for e in exps])
    except Exception as e:
        print("ERREUR : Impossible de joindre le serveur MLflow :", e)
        return
    # Si l'expérience n'existe pas, on la crée ; sinon on récupère son ID
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    # On s'assure que MLflow “travaille” dans cette expérience
    mlflow.set_experiment(experiment_name)

    # -----------------------------------------
    # 5) Lancement du run H2O AutoML
    # -----------------------------------------
    print("→ Je m’apprête à démarrer un run MLflow pour l’expérience:", experiment_name)
    with mlflow.start_run() as run:
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
