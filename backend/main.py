import pandas as pd
import io
import h2o

from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from utils.data_processing import separate_id_col, preprocess_for_model

app = FastAPI()

# Lancer H2O
h2o.init()

# Accéder à MLflow et charger le meilleur modèle
client = MlflowClient()
all_exps = [exp.experiment_id for exp in client.list_experiments()]
runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ALL)
run_id = runs.loc[runs['metrics.log_loss'].idxmin()]['run_id']
exp_id = runs.loc[runs['metrics.log_loss'].idxmin()]['experiment_id']
print(f'[+] Loading best model: Run {run_id} of Experiment {exp_id}')
best_model = mlflow.h2o.load_model(f"mlruns/{exp_id}/{run_id}/artifacts/model/")

@app.post("/predict")
async def predict(file: bytes = File(...)):
    try:
        print('[+] Prediction started')

        # Lire le fichier CSV
        file_obj = io.BytesIO(file)
        test_df = pd.read_csv(file_obj)

        # Appliquer les mêmes transformations qu’en entraînement
        test_df = preprocess_for_model(test_df)

        # Convertir en H2OFrame
        test_h2o = h2o.H2OFrame(test_df)

        # Séparer la colonne id si présente
        id_name, X_id, X_h2o = separate_id_col(test_h2o)

        # Prédiction
        preds = best_model.predict(X_h2o)

        # Résultat
        if id_name is not None:
            preds_list = preds.as_data_frame()['predict'].tolist()
            id_list = X_id.as_data_frame()[id_name].tolist()
            preds_final = dict(zip(id_list, preds_list))
        else:
            preds_final = preds.as_data_frame()['predict'].tolist()

        return JSONResponse(content=jsonable_encoder(preds_final))

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def home():
    return HTMLResponse(content="""
    <h2>API AutoML – Assurance Cross-Sell</h2>
    <p>Envoyez un fichier CSV via <code>/predict</code></p>
    <p>Interface Swagger disponible sur <code>/docs</code></p>
    """)
