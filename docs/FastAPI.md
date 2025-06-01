# FastAPI - Guide Complet

## Qu'est-ce que FastAPI ?
FastAPI est un framework web moderne, rapide et facile à utiliser pour construire des APIs avec Python. C'est comme un serveur de restaurant très efficace qui prend les commandes et les transmet rapidement à la cuisine.

## Fonctionnalités Principales

### 1. Performance
- **Vitesse** : Un des frameworks Python les plus rapides
- **Asynchrone** : Support natif de l'asyncio
- **Concurrent** : Gère plusieurs requêtes simultanément
- **Efficace** : Optimisé pour les performances

### 2. Facilité d'Utilisation
- **Documentation Automatique** : Génère la documentation OpenAPI
- **Validation des Données** : Vérification automatique des types
- **Éditeur Intelligent** : Support complet de l'autocomplétion
- **API Interactive** : Interface Swagger UI intégrée

### 3. Dans Notre Projet
- **Rôle** : Backend de l'application
- **Port** : 8000
- **Fonctionnalités** :
  * Gestion des prédictions
  * Communication avec H2O
  * API RESTful
  * Documentation automatique

## Installation et Configuration

### 1. Prérequis
```bash
# Installation de FastAPI et ses dépendances
pip install fastapi uvicorn
```

### 2. Structure de Base
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    data: dict

@app.post("/predict")
async def predict(request: PredictionRequest):
    return {"prediction": "result"}
```

### 3. Configuration dans Docker
```yaml
ports:
  - "8000:8000"
```

## Utilisation de Base

### 1. Création d'une API
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Modèle de données
class InsuranceData(BaseModel):
    age: int
    income: float
    has_insurance: bool

# Endpoint
@app.post("/predict")
async def predict_insurance(data: InsuranceData):
    try:
        # Logique de prédiction
        return {"probability": 0.75}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. Validation des Données
```python
from pydantic import BaseModel, Field

class CustomerData(BaseModel):
    age: int = Field(..., ge=0, le=120)
    income: float = Field(..., gt=0)
    has_insurance: bool
```

### 3. Documentation Automatique
```python
@app.get("/", tags=["Root"])
async def root():
    """
    Point d'entrée de l'API
    """
    return {"message": "Bienvenue sur l'API d'assurance"}
```

## Bonnes Pratiques

### 1. Structure du Projet
- Séparer les routes
- Utiliser des modèles Pydantic
- Implémenter la gestion d'erreurs
- Ajouter des tests

### 2. Sécurité
- Utiliser l'authentification
- Valider les entrées
- Gérer les CORS
- Implémenter le rate limiting

### 3. Performance
- Utiliser l'asyncio
- Mettre en cache les résultats
- Optimiser les requêtes
- Gérer la concurrence

## Dépannage Courant

### 1. Problèmes de Connexion
- Vérifier le port
- Vérifier les CORS
- Vérifier les permissions
- Vérifier les logs

### 2. Problèmes de Performance
- Optimiser les requêtes
- Utiliser le caching
- Gérer la mémoire
- Monitorer les ressources

### 3. Problèmes de Validation
- Vérifier les modèles
- Vérifier les types
- Vérifier les contraintes
- Vérifier les erreurs

## Ressources Utiles

### 1. Documentation Officielle
- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Guide de Démarrage](https://fastapi.tiangolo.com/tutorial/)
- [API Reference](https://fastapi.tiangolo.com/reference/)

### 2. Communautés
- [GitHub](https://github.com/tiangolo/fastapi)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/fastapi)
- [Discord](https://discord.gg/VQjSZaeJmf)

### 3. Tutoriels
- [Tutoriels Vidéo](https://www.youtube.com/c/Tiangolo)
- [Exemples de Code](https://github.com/tiangolo/fastapi/tree/master/docs_src)
- [Cours en Ligne](https://www.udemy.com/course/fastapi-course/) 