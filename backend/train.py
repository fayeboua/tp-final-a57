import os
import argparse
import h2o
from h2o.automl import H2OAutoML, get_leaderboard
import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
import json
import pandas as pd
from mlflow.models.signature import infer_signature
import mlflow.openai
from openai import OpenAI
import platform
import psutil

mlflow.openai.autolog()

client = OpenAI(api_key=os.environ.get("OPEN_AI_KEY", ""))

mlflow.set_tracking_uri("http://mlflow:5001")

def log_system_info():
    mlflow.log_param("platform", platform.platform())
    mlflow.log_param("processor", platform.processor())
    mlflow.log_param("cpu_count", psutil.cpu_count())
    mlflow.log_param("memory_total_GB", round(psutil.virtual_memory().total / (1024 ** 3), 2))

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

    mlflow_client = MlflowClient()

    try:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location="file:///app/backend/mlruns"
        )
    except Exception:
        experiment = mlflow_client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise RuntimeError(f"Impossible de trouver ou créer l'expérience '{experiment_name}'")
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    print("→ Lancement du run MLflow pour l’expérience:", experiment_name)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print("→ Run ID :", run_id)

        mlflow.set_tag("source", "h2o-automl-train")
        log_system_info()

        mlflow.log_param("max_models", models)
        mlflow.log_param("target", target)

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

        metrics = {
            "logloss": float(perf.logloss()),
            "AUC": float(perf.auc()),
            "rmse": float(perf.rmse()),
            "mse": float(perf.mse())
        }

        accuracy_list = perf.accuracy()
        if isinstance(accuracy_list, list) and len(accuracy_list) > 0:
            metrics["accuracy"] = float(accuracy_list[0][1])

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        temp_dir = "mlruns/tmp/mlflow_metrics"
        os.makedirs(temp_dir, exist_ok=True)
        metrics_path = os.path.join(temp_dir, f"metrics_{run_id}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path, artifact_path="metrics")

        pd_predictors = main_frame[predictors].as_data_frame()
        pd_preds = leader.predict(main_frame[predictors]).as_data_frame()
        signature = infer_signature(pd_predictors, pd_preds)

        mlflow.h2o.log_model(leader, artifact_path="model", signature=signature)

        lb = get_leaderboard(aml, extra_columns='ALL')
        df_leaderboard = lb.as_data_frame()
        temp_dir_lb = "mlruns/tmp/mlflow_leaderboard"
        os.makedirs(temp_dir_lb, exist_ok=True)
        leaderboard_csv = os.path.join(temp_dir_lb, f"leaderboard_{run_id}.csv")
        df_leaderboard.to_csv(leaderboard_csv, index=False)
        mlflow.log_artifact(leaderboard_csv, artifact_path="leaderboard")

        print(f"[✓] Modèle, métriques et leaderboard loggés dans MLflow (run_id={run_id})")

        try:
            prompt = f"""Voici les résultats d’un entraînement AutoML avec H2O :
                - Modèle leader : {leader.algo}
                - Métriques : {json.dumps(metrics, indent=2)}
                - Top 3 du leaderboard :\n{df_leaderboard.head(3).to_string(index=False)}

                Peux-tu résumer les performances du modèle et proposer des pistes d’amélioration ou d’analyse ?"""

            mlflow.set_tag("openai_summary", "in_progress")

            def generate_summary(model_name):
                return client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=500
                )

            try:
                completion = generate_summary("gpt-4o")
            except Exception:
                print("→ GPT-4o indisponible, tentative avec gpt-3.5-turbo…")
                completion = generate_summary("gpt-3.5-turbo")

            summary_text = completion.choices[0].message.content.strip()
            print("Résumé généré par OpenAI :\n", summary_text)

            summary_dir = "mlruns/tmp/mlflow_openai"
            os.makedirs(summary_dir, exist_ok=True)
            summary_path = os.path.join(summary_dir, f"openai_summary_{run_id}.txt")
            with open(summary_path, "w") as f:
                f.write(summary_text)

            mlflow.log_artifact(summary_path, artifact_path="openai_summary")
            mlflow.set_tag("openai_summary", "done")

        except Exception as e:
            print("Erreur OpenAI :", e)
            mlflow.set_tag("openai_summary", "failed")

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
