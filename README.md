# Prédiction de churn bancaire — Pipeline ETL + XGBoost

Projet complet de A à Z : génération des données, pipeline de nettoyage,
feature engineering et modèle de machine learning pour prédire les clients
qui vont quitter une banque.

## Ce que j'ai fait

- Génération d'un dataset bancaire fictif de 10 000 clients
- Pipeline ETL : nettoyage, gestion des valeurs manquantes, encodage
- Feature engineering : création de variables métier pertinentes
- Modèle XGBoost avec optimisation des hyperparamètres
- Évaluation : 81,2% de précision, matrice de confusion, feature importance

## Stack

Python (pandas, numpy, scikit-learn, xgboost) · Airflow · Docker

## Ce que j'ai trouvé — et pourquoi ça compte

Sans modèle prédictif, une banque détecte le churn **après** que le client
soit parti — trop tard pour agir.

L'analyse révèle ce que l'oeil nu ne voit pas :
le nombre de produits souscrits est la variable la plus prédictive du départ,
devant le salaire et le solde. Un client avec 1 seul produit a une probabilité
de churn bien supérieure à un client multi-produits — même si son solde est élevé.

Ce résultat contre-intuitif est clé : une banque qui ciblerait ses actions
de rétention uniquement sur les gros soldes passerait à côté des vrais clients
à risque. Le modèle permet de prioriser les appels commerciaux sur les clients
1 produit + solde > 50 000€ — le segment le plus à risque identifié.

Le modèle souffre d'un déséquilibre des classes (80/20) — piste d'amélioration
identifiée : appliquer SMOTE pour mieux détecter les churners minoritaires.

## Lancer le projet

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
python pipeline/etl.py
python model/train_model.py
```
