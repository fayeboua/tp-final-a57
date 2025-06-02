# Exemple de Dockerfile pour le backend (backend/Dockerfile)

FROM python:3.9-slim

# 1) Installer Java et les dépendances système pour psycopg2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       default-jre \
       gcc \
       libpq-dev \
       python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Copier et installer les requirements Python (y compris psycopg2-binary si nécessaire)
COPY requirements-backend.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt psycopg2-binary

# 3) Copier le code de votre API/backend dans le conteneur
COPY . /app

# 4) Exposer le port de l’API (FastAPI/Uvicorn)
EXPOSE 8000

# 5) Commande de lancement du serveur Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
