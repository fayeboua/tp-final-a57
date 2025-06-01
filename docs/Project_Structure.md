# Structure du Projet - Guide Complet

## Vue d'Ensemble
Notre projet est structuré comme un restaurant moderne, où chaque composant a un rôle spécifique :
- Backend (FastAPI) : La cuisine
- Frontend (Streamlit) : La salle
- H2O (AutoML) : Le chef
- MLflow : Le livre de recettes

## Structure des Dossiers

### 1. Dossier Principal
```
End-to-End-AutoML-Insurance/
├── backend/           # Service FastAPI
├── frontend/         # Application Streamlit
├── docs/            # Documentation
├── data/            # Données
└── docker-compose.yml
```

### 2. Backend (FastAPI)
```
backend/
├── app/
│   ├── main.py      # Point d'entrée
│   ├── models/      # Modèles de données
│   ├── routes/      # Routes API
│   └── services/    # Services métier
├── mlruns/          # Dossier MLflow
└── Dockerfile
```

### 3. Frontend (Streamlit)
```
frontend/
├── app.py           # Application principale
├── pages/          # Pages additionnelles
├── components/     # Composants réutilisables
└── Dockerfile
```

## Configuration Docker

### 1. Services
- **Backend** : Port 8000
- **Frontend** : Port 8501
- **MLflow** : Port 5000

### 2. Volumes
- `mlruns_data` : Données MLflow
- `model_data` : Modèles sauvegardés

### 3. Réseau
- Réseau dédié : `insurance_network`
- Communication inter-services

## Flux de Données

### 1. Upload de Données
1. Interface Streamlit
2. Validation des données
3. Stockage temporaire
4. Traitement H2O

### 2. Prédiction
1. Requête API
2. Chargement modèle
3. Prédiction
4. Retour résultat

### 3. Suivi
1. Enregistrement MLflow
2. Sauvegarde modèle
3. Métriques
4. Visualisation

## Bonnes Pratiques

### 1. Organisation
- Séparation claire des responsabilités
- Documentation à jour
- Tests unitaires
- Gestion des versions

### 2. Développement
- Standards de code
- Revue de code
- Tests automatisés
- Intégration continue

### 3. Déploiement
- Environnements séparés
- Gestion des secrets
- Monitoring
- Backup

## Dépannage

### 1. Problèmes Courants
- Connexion entre services
- Accès aux données
- Performance
- Erreurs de build

### 2. Solutions
- Vérification des logs
- Test des services
- Validation des données
- Mise à jour des dépendances

## Ressources

### 1. Documentation
- [Guide d'Installation](guide_ultra_detaile.txt)
- [Documentation H2O](docs/H2O.md)
- [Documentation FastAPI](docs/FastAPI.md)
- [Documentation Streamlit](docs/Streamlit.md)
- [Documentation MLflow](docs/MLflow.md)

### 2. Support
- Issues GitHub
- Documentation officielle
- Forums communautaires
- Stack Overflow 