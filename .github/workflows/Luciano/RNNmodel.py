# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 07:08:43 2024

@author: langh
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import re
import os
from datetime import timedelta
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Définir le répertoire
nouveau_repertoire = r'C:\Users\langh\OneDrive\Desktop\DATA Sc'
os.chdir(nouveau_repertoire)

# Charger les données
df_c = df_knn.copy()
df_base = pd.read_csv("weatherAUS.csv")
df_date = df_base[df_base.index.isin(df.index)]["Date"]
df_date = pd.to_datetime(df_date)




### Prediction J+1
villes = []
resultat_ville = []

for ville in df_base.Location.unique():
    df = df_c[df_c[ville]>0]
    df['Date'] = pd.to_datetime(df_date)

    def creer_fenetres(df, taille_fenetre=8, fenetre_prediction=1, colonne_cible="RainTomorrow"):
        X, y = [], []
        for i in range(len(df) - taille_fenetre - fenetre_prediction + 1):
            fenetre_df = df.iloc[i:(i + taille_fenetre + fenetre_prediction)]
            fenetre_df['Date'] = pd.to_datetime(fenetre_df['Date'])
            toutes_consecutives = all(fenetre_df['Date'].iloc[j] == fenetre_df['Date'].iloc[0] + timedelta(days=j) for j in range(taille_fenetre + fenetre_prediction))
            if not toutes_consecutives:
                continue
            donnees_fenetre = fenetre_df.iloc[:taille_fenetre, :-1].values
            valeur_cible = fenetre_df.iloc[taille_fenetre:taille_fenetre + fenetre_prediction][colonne_cible].values
            X.append(donnees_fenetre)
            y.append(valeur_cible)
        return np.array(X), np.array(y)

    X, y = creer_fenetres(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(LSTM(units=350, return_sequences=True, input_shape=(8, X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=350))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    
    learning_rate = 0.002
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=7, batch_size=50)
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Test')
    plt.title('Précision en fonction du nombre d\'époques')
    plt.xlabel('Épochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    
    perte, precision = model.evaluate(X_test, y_test)
    print(f"{ville} Perte : {perte}, Précision : {precision}")
    villes.append(ville)
    resultat_ville.append(precision)



### Prediction J+5
villes = []
resultat_ville_j1_precision = []
resultat_ville_j1_recall = []
resultat_ville_j1_f1 = []
resultat_ville_j2_precision = []
resultat_ville_j2_recall = []
resultat_ville_j2_f1 = []
resultat_ville_j3_precision = []
resultat_ville_j3_recall = []
resultat_ville_j3_f1 = []
resultat_ville_j4_precision = []
resultat_ville_j4_recall = []
resultat_ville_j4_f1 = []
resultat_ville_j5_precision = []
resultat_ville_j5_recall = []
resultat_ville_j5_f1 = []


for ville in df_base.Location.unique():
    df = df_c[df_c[ville] > 0]
    df['Date'] = pd.to_datetime(df_date)

    def creer_fenetres(df, taille_fenetre=5, fenetre_prediction=5, colonne_cible="RainTomorrow"):
        X, y = [], []
        for i in range(len(df) - taille_fenetre - fenetre_prediction + 1):
            fenetre_df = df.iloc[i:(i + taille_fenetre + fenetre_prediction)]
            fenetre_df['Date'] = pd.to_datetime(fenetre_df['Date'])
            toutes_consecutives = all(fenetre_df['Date'].iloc[j] == fenetre_df['Date'].iloc[0] + timedelta(days=j) for j in range(taille_fenetre + fenetre_prediction))
            if not toutes_consecutives:
                continue
            donnees_fenetre = fenetre_df.iloc[:taille_fenetre, :-1].values
            valeur_cible = fenetre_df.iloc[taille_fenetre:taille_fenetre + fenetre_prediction][colonne_cible].values
            X.append(donnees_fenetre)
            y.append(valeur_cible)
        return np.array(X), np.array(y)

    X, y = creer_fenetres(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(LSTM(units=500, return_sequences=True, input_shape=(5, X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=500))
    model.add(Dropout(0.2))
    model.add(Dense(units=5, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=50)
    y_pred = model.predict(X_test)
    for i in range(5):
       predictions = y_pred[:, i] > 0.5
       jour_precision = accuracy_score(y_test[:, i], predictions)
       jour_recall = recall_score(y_test[:, i], predictions)
       jour_f1 = f1_score(y_test[:, i], predictions)

       if i == 0:
           resultat_ville_j1_precision.append(jour_precision)
           resultat_ville_j1_recall.append(jour_recall)
           resultat_ville_j1_f1.append(jour_f1)
       elif i == 1:
           resultat_ville_j2_precision.append(jour_precision)
           resultat_ville_j2_recall.append(jour_recall)
           resultat_ville_j2_f1.append(jour_f1)
       elif i == 2:
           resultat_ville_j3_precision.append(jour_precision)
           resultat_ville_j3_recall.append(jour_recall)
           resultat_ville_j3_f1.append(jour_f1)
       elif i == 3:
           resultat_ville_j4_precision.append(jour_precision)
           resultat_ville_j4_recall.append(jour_recall)
           resultat_ville_j4_f1.append(jour_f1)
       elif i == 4:
           resultat_ville_j5_precision.append(jour_precision)
           resultat_ville_j5_recall.append(jour_recall)
           resultat_ville_j5_f1.append(jour_f1)
