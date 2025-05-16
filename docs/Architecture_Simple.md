# Architecture Simple - Pour les Débutants

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
