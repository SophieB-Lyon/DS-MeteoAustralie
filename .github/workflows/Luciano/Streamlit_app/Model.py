# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 10:40:49 2024

@author: langh
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
from xgboost import XGBClassifier
import joblib
import json

# Charger les données
df_met = pd.read_csv('data_basique.csv')
df_met.drop("Date", inplace=True, axis=1)
cols_supp = [col for col in df_met.columns if "WindGust" in col or "WindDir9am" in col]
df_met.drop(columns=cols_supp, inplace=True)

X = df_met.drop('RainTomorrow', axis=1)
y = df_met['RainTomorrow'].astype(int)

# Diviser en Trai/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Parametres pour GridSearchCV
param_grid = {
    'n_estimators': [40, 80, 120],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1]
}

# Initialiser le model
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Évaluer le modèle
meilleur_modele = grid_search.best_estimator_
predictions = meilleur_modele.predict(X_test_scaled)
exactitude = accuracy_score(y_test, predictions)
rappel = recall_score(y_test, predictions)

# Save le modèle et le scaler (à importer en streamlit)
joblib.dump(meilleur_modele, 'xgb_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Save les hiperparametres (à importer en streamlit))
hyperparametres = {
    'n_estimators': meilleur_modele.n_estimators,
    'max_depth': meilleur_modele.max_depth,
    'learning_rate': meilleur_modele.learning_rate
}

info_modele = {
    'algorithme': 'XGBoost',
    'hyperparametres': hyperparametres,
    'exactitude': exactitude,
    'rappel': rappel
}

with open('modele_info.json', 'w') as fichier:
    json.dump(info_modele, fichier)
