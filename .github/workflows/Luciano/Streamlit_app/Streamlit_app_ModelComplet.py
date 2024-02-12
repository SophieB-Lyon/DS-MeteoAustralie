# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 09:37:03 2024

@author: langh
"""

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import numpy as np
import joblib
import json

# Title
st.title("Meteo d'Australie")

# Description
st.markdown("""
    Cette application permet de prédire la pluie du jour suivant en choisissant une ville d'Australie et certaines variables météorologiques.
    Il suffit de sélectionner vos options et variables ci-dessous pour obtenir la prédiction.
""", unsafe_allow_html=True)

# Charger les données
df_met = pd.read_csv('data_basique.csv')
df_base = pd.read_csv('weatherAUS.csv')
df_met.drop("Date", inplace=True, axis=1)

# Suppression des colonnes pour simplifier
cols_supp = [col for col in df_met.columns if "WindGust" in col or "WindDir9am" in col]
df_met.drop(columns=cols_supp, inplace=True)

# S'assurer que df_met a les colonnes pour WindDir3pm
wind_directions = df_base['WindDir3pm'].dropna().unique()
for wind_dir in wind_directions:
    df_met[f'WindDir3pm_{wind_dir}'] = 0

# Sélection de l'utilité, ville, et wind direction
utilidad = st.selectbox('Sélectionnez l\'utilité :', ['Agriculteur', 'Hôtel Touristique'])
ville_sel = st.selectbox('Ville :', df_base['Location'].unique())
wind_dir_selected = st.selectbox('Direction du vent à 15h :', wind_directions)

# Initialisation des entrées dfor user
entrees = {}
stats = df_met.describe()
alerts = False
datos_completados = 0  # Compteur de données saisies by user

# Organiser les variables pour l'interface streamlit
input_order = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']  # Placer d'abord les températures dans un ordre spécifique
other_inputs = [col for col in df_met.columns if col not in input_order and 'Location_' not in col and 'WindDir3pm_' not in col and col != 'RainTomorrow']
ordered_inputs = input_order + other_inputs  # Combiner les listes pour garder l'ordre souhaité

def add_input(column, container):
    # Définir l'unités
    unit = " (km/h)" if "WindSpeed" in column else " (hPa)" if "Pressure" in column else " (mm)" if "Rainfall" in column else " (0/1)" if "RainToday" in column else " (°C)" if "Temp" in column else " (%)"
    # Capturer la valeur by user
    value = container.number_input(f"{column}{unit}:", format='%f', step=0.1, key=column)
    entrees[column] = value

# Diviser l'interface en 3 colonnes
col1, col2, col3 = st.columns(3)

# Assigner les entrées à chaque colonne
inputs_per_col = len(ordered_inputs) // 3
for i, col in enumerate(ordered_inputs):
    if i < inputs_per_col:
        add_input(col, col1)
    elif i < 2 * inputs_per_col:
        add_input(col, col2)
    else:
        add_input(col, col3)

# Vérifier le qty des variable saisies
datos_completados = sum([1 for value in entrees.values() if value > 0])

if datos_completados < 5:
    st.warning("Au moins 5 données sont nécessaires pour exécuter la prédiction.")
    ejecutar_prediccion = False
else:
    ejecutar_prediccion = True

# Bouton de confirmation si nécessaire
if alerts:
    confirmation = st.button('Confirmer les valeurs et prédire')
else:
    confirmation = True

# Prédiction après vérification ou confirmation
if (st.button('Prédire si demain il va pleuvoir') or confirmation) and ejecutar_prediccion:
    if alerts and not confirmation:
        st.error("Veuillez confirmer les valeurs hors du range pour continuer.")
    else:
        # Préparer df_usr pour la prédiction
        df_usr = pd.DataFrame(columns=df_met.columns.drop('RainTomorrow'))
        df_usr.loc[0] = np.nan

        # Remplir df_usr avec les entrées de l'utilisateur et mettre toutes les colonnes 'Location_' à 0
        for col in df_met.columns.drop('RainTomorrow'):
            if col.startswith('Location_'):
                df_usr[col] = 0  
            elif col.startswith('WindDir3pm_'):
                df_usr[col] = 0
            else:
                df_usr.at[0, col] = entrees.get(col, np.nan)
        
        # Mettre la direction du vent sélectionnée à 1
        df_usr[f'WindDir3pm_{wind_dir_selected}'] = 1

        # Mettre la colonne de la ville sélectionnée à 1
        df_usr[f'Location_{ville_sel}'] = 1
            
        # # Afficher DataFrame 
        # st.write("Données d'entrée avant KNNimputer :")
        # st.dataframe(df_usr)        
        
        imputer = KNNImputer(n_neighbors=5)
        df_met_imputed = imputer.fit(df_met.drop(columns=['RainTomorrow']))
        df_usr = df_usr.reindex(columns=df_met.drop(columns=['RainTomorrow']).columns)

        # Transformer et convertir le résultat en DataFrame tout en gardant les noms des colonnes
        df_usr_imputed_array = imputer.transform(df_usr)
        df_usr = pd.DataFrame(df_usr_imputed_array, columns=df_usr.columns)
        
        # # Afficher DataFrame
        # st.write("Données d'entrée de l'utilisateur après KNNimputer :")
        # st.dataframe(df_usr)

        # Scaler pour les donnes by user
        scaler = StandardScaler()
        scaler.fit(df_met.drop(columns=['RainTomorrow']))
        df_usr_scaled = scaler.transform(df_usr)
        
        df_usr[:] = df_usr_scaled

        # # Afficher le DataFrame
        # st.write("Données d'entrée de l'utilisateur après KNN et Scaler :")
        # st.dataframe(df_usr)

        # Charger le model
        model = joblib.load('xgb_model.joblib')

        # Réaliser la prédiction
        pred_proba = model.predict_proba(df_usr)[:, 1]  
        # Seuilbasé sur le choix de l'utilisateur
        threshold = 0.25 if utilidad == 'Agriculteur' else 0.5
        pred_usr = (pred_proba >= threshold).astype(int)
        res = "Pluie demain. Prenez le parapluie !" if pred_usr[0] == 1 else "Pas de pluie. Laissez le parapluie !"
        st.markdown(f'**{res}**', unsafe_allow_html=True)

        # Afficher l'image 
        img = 'Parapluie_ouvert.png' if pred_usr[0] == 1 else 'Parapluie_ferme.png'
        st.image(img, width=250)
        
        if st.button('Voir les détails du modèle'):           
            try:
                with open('modelo_info.json', 'r') as file:
                    modelo_info = json.load(file)
                st.write(f"Algorithme utilisé : {modelo_info['algoritmo']}")
                st.write(f"Hyperparamètres du modèle : {modelo_info['hiperparametros']}")
                st.write(f"Précision du jeu de test : {modelo_info['accuracy']:.2f}%")
                st.write(f"Recall du jeu de test : {modelo_info['recall']:.2f}%")
            except FileNotFoundError:
                st.error("Fichier d'informations du modèle non trouvé.")
