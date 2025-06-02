# Dockerfile pour le backend (FastAPI + MLflow + H2O)

# 1) Image de base
FROM python:3.9

# 2) Créer un utilisateur non-root (UID=1000/GID=1000)
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# 3) Définir le répertoire de travail
WORKDIR /app

# 4) Copier et installer uniquement les dépendances
COPY requirements-backend.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5) Installer Java (JRE) pour H2O
RUN apt-get update \
 && apt-get install -y default-jre \
 && rm -rf /var/lib/apt/lists/*

# 6) Copier tout le code de votre application
COPY . /app

# 7) Créer le dossier mlruns si nécessaire et donner la propriété à appuser
RUN mkdir -p /app/backend/mlruns \
 && chown -R appuser:appuser /app/backend/mlruns

# 8) Donner la propriété complète du dossier /app à appuser
RUN chown -R appuser:appuser /app

# 9) Basculer vers l’utilisateur non-root
USER appuser

# 10) Exposer le port Uvicorn
EXPOSE 8000

# 11) Commande de démarrage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
