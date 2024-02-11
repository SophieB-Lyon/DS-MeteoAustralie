# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:04:40 2024

@author: langh
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np

# Chargement des données
df_met = pd.read_csv('data_basique.csv')
df_base = pd.read_csv('weatherAUS.csv')
df_met.drop("Date", inplace=True, axis=1)

# Suppression des colonnes pour simplifier
cols_supp = [col for col in df_met.columns if "WindGust" in col or "WindDir" in col]
df_met.dro# Sélection de la ville
villes = df_base['Location'].unique()
ville_sel = st.selectbox('Ville:', villes)p(columns=cols_supp, inplace=True)



# Initialisation des entrées utilisateur
entrees = {}
stats = df_met.describe()
alerts = False

# Entrées utilisateur + vérification du range
for col in df_met.columns:
    if 'Location_' in col or col == 'RainTomorrow':
        continue
    unit = " (hPa)" if "Pressure" in col else " (mm)" if "Rainfall" in col else " (0/1)" if "RainToday" in col else " (°C)" if "Temp" in col else " (%)" if "Humidity" in col else " (km/h)"
    val = st.number_input(f'{col}{unit}:', format='%f', step=0.1, key=col)

    # Vérification du range
    if val:
        moy = stats[col]['mean']
        ecart = stats[col]['std']
        inf = moy - 2 * ecart
        sup = moy + 2 * ecart
        if not (inf <= val <= sup):
            st.warning(f'Merci de verifier {col} ({val}){unit}, il semble légèrement hors du range habituel.')
            alerts = True

# Bouton de confirmation si nécessaire
if alerts:
    confirmation = st.button('Confirmer les valeurs et prédire')
else:
    confirmation = True

# Prédiction post vérification ou confirmation
if st.button('Prédire si demain il va pleuvoir') or confirmation:
    if alerts and not confirmation:
        st.error("Veuillez confirmer les valeurs hors du range pour continuer.")
    else:
        df_usr = pd.DataFrame([entrees])
        
        # Assurer le même format que df_met
        for col in df_met.columns.drop('RainTomorrow'):
            if col not in df_usr.columns:
                df_usr[col] = 0
        df_usr = df_usr.reindex(columns=df_met.columns.drop('RainTomorrow'))
        
        # Prétraitement + mise à l'échelle
        scaler = StandardScaler()
        df_usr_scaled = scaler.fit_transform(df_usr.select_dtypes(include=np.number))
        df_usr[df_usr.select_dtypes(include=np.number).columns] = df_usr_scaled

        X = df_met.drop('RainTomorrow', axis=1)
        y = df_met['RainTomorrow'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {'n_estimators': [25, 50], 'max_depth': [3, 5], 'learning_rate': [0.1]}
        mod = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), params, cv=4)
        mod.fit(X_train, y_train)

        preds = mod.predict(X_test)
        prec = accuracy_score(y_test, preds) * 100
        
        pred_usr = mod.predict(df_usr)
        res = "Pluie demain. Prenez le parapluie !" if pred_usr[0] else "Pas de pluie. Laissez le parapluie !"
        st.markdown(f'**{res}**', unsafe_allow_html=True)

        img = 'Parapluie_ouvert.png' if pred_usr[0] else 'Parapluie_ferme.png'
        st.image(img, width=250)

        st.write("Méthode ML utilisée : XGBClassifier")
        st.write(f"Hyperparamètres : {mod.best_params_}")
        st.write(f"Précision du modèle : {prec:.2f}%")
