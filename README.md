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

Le modèle atteint 87% de précision sur le jeu de test. 
Les 3 variables les plus prédictives du churn sont le solde du compte, 
l'ancienneté client et le nombre de produits souscrits. 
Les clients avec un solde élevé mais peu de produits sont les plus à risque.

## Lancer le projet

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
python pipeline/etl.py
python model/train_model.py
```
