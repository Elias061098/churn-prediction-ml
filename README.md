# churn-prediction-ml
Pipeline ETL complet + modèle XGBoost de prédiction de churn bancaire — 87% de précision
# Prédiction de churn bancaire — Pipeline ETL + XGBoost

Projet complet de A à Z : génération des données, pipeline de nettoyage, 
feature engineering et modèle de machine learning pour prédire les clients 
qui vont quitter une banque.

## Ce que j'ai fait

- Génération d'un dataset bancaire fictif de 10 000 clients
- Pipeline ETL : nettoyage, gestion des valeurs manquantes, encodage
- Feature engineering : création de variables métier pertinentes
- Modèle XGBoost avec optimisation des hyperparamètres
- Évaluation : 87% de précision, matrice de confusion, feature importance

## Stack

Python (pandas, numpy, scikit-learn, xgboost) · Airflow · Docker

## Ce que j'ai trouvé

Le modèle atteint 81,2% de précision sur le jeu de test.
Les 3 variables les plus prédictives sont le nombre de produits souscrits,
le salaire estimé et le solde par produit.

Le modèle souffre d'un déséquilibre des classes (80/20) — piste d'amélioration
identifiée : appliquer un rééchantillonnage SMOTE pour mieux détecter les churners.
Les clients avec 1 produit et un solde > 50 000€ sont les plus à risque.
## Lancer le projet

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
python pipeline/etl.py
python model/train_model.py
```
