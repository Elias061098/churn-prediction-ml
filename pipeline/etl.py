import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

# Génération des données brutes
df = pd.DataFrame({
    'client_id': range(1, n + 1),
    'age': np.random.randint(18, 75, n),
    'anciennete_ans': np.random.randint(0, 15, n),
    'solde': np.random.normal(50000, 30000, n).clip(0, 200000).round(2),
    'nb_produits': np.random.choice([1, 2, 3, 4], n, p=[0.45, 0.35, 0.15, 0.05]),
    'carte_credit': np.random.choice([0, 1], n, p=[0.3, 0.7]),
    'membre_actif': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    'salaire_estime': np.random.normal(60000, 25000, n).clip(15000, 150000).round(2),
    'pays': np.random.choice(['France', 'Allemagne', 'Espagne'], n, p=[0.5, 0.25, 0.25]),
    'churn': np.random.choice([0, 1], n, p=[0.8, 0.2]),
})

# Injection de valeurs manquantes pour simuler des données réelles
df.loc[np.random.choice(df.index, 200), 'solde'] = np.nan
df.loc[np.random.choice(df.index, 150), 'salaire_estime'] = np.nan

print(f"Dataset brut : {df.shape}")
print(f"Valeurs manquantes :\n{df.isnull().sum()}")

# ── Nettoyage
df['solde'] = df['solde'].fillna(df['solde'].median())
df['salaire_estime'] = df['salaire_estime'].fillna(df['salaire_estime'].median())

# ── Feature engineering
df['solde_par_produit'] = (df['solde'] / df['nb_produits']).round(2)
df['ratio_salaire_solde'] = (df['solde'] / df['salaire_estime']).round(3)
df['client_premium'] = ((df['solde'] > 100000) & (df['nb_produits'] >= 2)).astype(int)

# ── Encodage
df = pd.get_dummies(df, columns=['pays'], drop_first=True)

df.to_csv('pipeline/clients_clean.csv', index=False)
print(f"\nDataset nettoyé : {df.shape}")
print(df.head())
