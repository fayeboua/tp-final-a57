
# Analyse de l'image — Pipeline d’IA 

![image](https://github.com/user-attachments/assets/03a32fc0-6b8a-4e52-ab86-0cee46587475)


L’image montre un **pipeline de bout en bout** pour une application de machine learning en production. Ce pipeline est découpé en **deux parties principales** : le **backend** et le **frontend**.



### 1. Description des composants

| Composant      | Rôle dans le système                                                                                                                                                                                                                            |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **H2O AutoML** | Moteur d’apprentissage automatique. Il entraîne automatiquement plusieurs modèles de ML à partir des données fournies. L’utilisateur n’a pas besoin de spécifier l’algorithme : H2O essaie différentes approches et sélectionne les meilleures. |
| **MLflow**     | Outil de suivi des expérimentations. Il enregistre les performances des modèles produits par H2O AutoML, garde une trace des métriques, des hyperparamètres et des artefacts, et sert de registre de modèles (model registry).                  |
| **FastAPI**    | API REST rapide qui sert de **point d’accès aux modèles**. Elle permet à des applications externes d’envoyer des requêtes (ex : prédictions), en appelant les modèles stockés via MLflow.                                                       |
| **Streamlit**  | Interface utilisateur graphique. Il s'agit d'une application web accessible aux utilisateurs finaux, leur permettant de téléverser des fichiers ou de faire des prédictions via l’API FastAPI.                                                  |



### 2. Interaction de l'utilisateur avec le système

* L’utilisateur interagit uniquement avec **Streamlit** (le frontend).
* Streamlit collecte les entrées utilisateur (ex : un fichier CSV ou des champs remplis manuellement).
* Il envoie une **requête API** (généralement en JSON) vers **FastAPI**.
* FastAPI reçoit cette requête, appelle le **modèle enregistré dans MLflow**, effectue une prédiction, et retourne les résultats.
* Streamlit affiche les **résultats à l’utilisateur**, généralement sous forme de tableau ou graphique.

**Résumé de l’échange :**

```
User → Streamlit → FastAPI → MLflow + modèle H2O → FastAPI → Streamlit → User
```



### 3. Étapes principales du flux de travail

1. **Collecte des données** : données clients, logs, formulaires, fichiers CSV, etc.
2. **Exploration & prétraitement** :

   * Nettoyage des données.
   * Feature engineering.
   * Anonymisation si nécessaire.
3. **Entraînement automatique (H2O AutoML)** :

   * Plusieurs modèles sont générés et comparés.
   * Le meilleur modèle est sélectionné.
4. **Suivi & traçabilité (MLflow)** :

   * Les métriques, modèles et hyperparamètres sont sauvegardés.
5. **Déploiement via FastAPI** :

   * Le meilleur modèle est exposé via un endpoint d’API.
6. **Interface utilisateur avec Streamlit** :

   * Application web légère permettant à un utilisateur d’interagir avec le système.
   * Affichage en temps réel des prédictions.
7. **Maintenance continue (MLOps)** :

   * Possibilité de mettre à jour les modèles.
   * Suivi des performances.
   * Analyse des erreurs.



### 4. Procédure générale de fonctionnement (interactions entre les composants)

1. **H2O AutoML** entraîne automatiquement les modèles à partir des données brutes ou préparées.
2. Les modèles sont enregistrés avec **MLflow**, qui sert aussi de dépôt de modèles (avec versionnement).
3. **FastAPI** se connecte à MLflow pour charger le modèle sélectionné, expose une route HTTP (`/predict`) accessible aux services externes.
4. **Streamlit**, qui est l’interface utilisateur, **envoie des données utilisateur (JSON, fichier CSV, etc.)** via cette route d’API.
5. FastAPI appelle le modèle avec les données et retourne la prédiction à Streamlit.
6. Streamlit **affiche les résultats à l’utilisateur final**.



### Synthèse finale (structure logique) :

| Couche      | Élément        | Fonction principale                         |
| ----------- | -------------- | ------------------------------------------- |
| Backend     | H2O AutoML     | Entraîne automatiquement des modèles        |
| Backend     | MLflow         | Sauvegarde et gère les modèles entraînés    |
| Backend     | FastAPI        | Sert les modèles via une API REST           |
| Frontend    | Streamlit      | Interface graphique pour l’utilisateur      |
| Utilisateur | Navigateur Web | Envoie une requête, visualise la prédiction |




# Annexe - Architecture 

# Notre Application est comme un Restaurant

```
┌─────────────────────────────────────────────────────────────────┐
│                        LE RESTAURANT                            │
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   STREAMLIT │     │   FASTAPI   │     │     H2O     │       │
│  │  (La Salle) │     │  (La Cuisine)│     │  (Le Chef)  │       │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘       │
│         │                   │                   │               │
│         │  "Je veux une     │  "Voici la       │  "Je vais     │
│         │   prédiction"     │   commande"       │   cuisiner"   │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  Interface  │     │    API      │     │  Modèles    │       │
│  │  Utilisateur│     │  Backend    │     │  AutoML     │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     MLFLOW                              │   │
│  │              (Le Livre de Recettes)                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

# Comment ça marche ?

1. **L'Utilisateur (Vous)**
   - Vous entrez dans le restaurant (ouvrez l'application)
   - Vous commandez un plat (demandez une prédiction)

2. **Streamlit (La Salle)**
   - Reçoit votre commande
   - Vous montre le menu (interface utilisateur)
   - Transmet votre commande à la cuisine

3. **FastAPI (La Cuisine)**
   - Reçoit la commande de la salle
   - Prépare les ingrédients (prépare les données)
   - Transmet la commande au chef

4. **H2O (Le Chef)**
   - Reçoit la commande de la cuisine
   - Utilise ses recettes (modèles)
   - Prépare le plat (fait la prédiction)

5. **MLflow (Le Livre de Recettes)**
   - Garde toutes les recettes (modèles)
   - Note les modifications (suivi des versions)
   - Aide le chef à s'améliorer (optimisation)

# Les Ports (Les Portes du Restaurant)

- **Streamlit** : Port 8501 (Porte d'entrée)
- **FastAPI** : Port 8000 (Porte de la cuisine)
- **H2O** : Port 54321 (Porte du chef)

# Les Données (Le Stock)

- **mlruns_data** : Où sont stockées les recettes
- **backend_data** : Où sont stockés les ingrédients
- **frontend_data** : Où sont stockés les menus

# La Sécurité (Les Gardes)

- Chaque service a ses propres gardes
- Les données sont protégées
- Seuls les services autorisés peuvent communiquer

# En Cas de Problème

1. **Le Restaurant ne s'ouvre pas ?**
   - Vérifiez que Docker est allumé
   - Vérifiez que tous les services sont démarrés

2. **La Commande ne passe pas ?**
   - Vérifiez que tous les ports sont ouverts
   - Vérifiez que les services communiquent

3. **Le Chef ne répond pas ?**
   - Vérifiez que H2O est bien démarré
   - Vérifiez que les modèles sont chargés

# Pour Résumer

- C'est comme un restaurant bien organisé
- Chaque partie a son rôle
- Tout est connecté et sécurisé
- Si quelque chose ne marche pas, on sait où chercher 
