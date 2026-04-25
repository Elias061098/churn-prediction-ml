import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

np.random.seed(42)
n = 10000

df = pd.DataFrame({
    'age': np.random.randint(18, 75, n),
    'anciennete_ans': np.random.randint(0, 15, n),
    'solde': np.random.normal(50000, 30000, n).clip(0, 200000).round(2),
    'nb_produits': np.random.choice([1, 2, 3, 4], n, p=[0.45, 0.35, 0.15, 0.05]),
    'carte_credit': np.random.choice([0, 1], n, p=[0.3, 0.7]),
    'membre_actif': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    'salaire_estime': np.random.normal(60000, 25000, n).clip(15000, 150000).round(2),
    'pays_Allemagne': np.random.choice([0, 1], n, p=[0.75, 0.25]),
    'pays_Espagne': np.random.choice([0, 1], n, p=[0.75, 0.25]),
    'solde_par_produit': np.random.normal(25000, 15000, n).clip(0, 100000).round(2),
    'ratio_salaire_solde': np.random.normal(0.8, 0.4, n).clip(0, 3).round(3),
    'client_premium': np.random.choice([0, 1], n, p=[0.85, 0.15]),
    'churn': np.random.choice([0, 1], n, p=[0.8, 0.2]),
})

X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Précision : {acc*100:.1f}%")
print(classification_report(y_test, y_pred))

# Matrice de confusion
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion')
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png', dpi=150)
plt.show()

# Feature importance
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=importance.values, y=importance.index, hue=importance.index, legend=False, palette='Blues_d')
plt.title('Feature Importance — XGBoost')
plt.tight_layout()
plt.savefig('model/feature_importance.png', dpi=150)
plt.show()

print(f"\nTop 3 variables prédictives :")
print(importance.head(3))
