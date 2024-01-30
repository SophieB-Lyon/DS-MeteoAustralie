# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 08:05:49 2023

@author: Usuario
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
#from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import GridSearchCV
#from xgboost import XGBClassifier
from collections import Counter
from sklearn.metrics import roc_curve, roc_auc_score
#import xgboost as xgb
import re
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Définir le répertoire
nouveau_repertoire = r'C:\Users\Usuario\Desktop\DATA Sc\Fil Rouge'
os.chdir(nouveau_repertoire)

# Importer les données
df_base = pd.read_csv("weatherAUS.csv")
df_date = df_base[df_base.index.isin(df.index)]["Date"]
df_date = pd.to_datetime(df_date)
df = df_knn.copy()

ville_cluster = pd.read_csv("climats.csv", index_col="Location")
ville_cluster.drop(columns=ville_cluster.columns[ville_cluster.columns.str.contains('Unnamed')], inplace=True)

columnas_ite = df.loc[:, "Adelaide":"Woomera"].columns
for i in df.index:
    for col in columnas_ite:
        if df.at[i, col] == 1:
            for ville in ville_cluster.index:
                if re.search(ville, col):
                    df.at[i, "Cluster"] = pd.to_numeric(ville_cluster[ville_cluster.index == ville].values[0], errors='coerce')

def train_test(df, col_target, test_size):
    data = df.drop(col_target, axis=1)
    target = df[col_target]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, stratify=target, random_state=77)
    return X_train, X_test, y_train, y_test

def entrenar_modelo_deep_learning(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate):
    modelo = Sequential()
    modelo.add(Dense(units=50, input_dim=X_train.shape[1], activation='relu'))
    modelo.add(Dense(units=50, activation='tanh'))
    modelo.add(Dense(units=1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    modelo.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=50, min_lr=0.0001)
    class TimeHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []
        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = time.time()
        def on_epoch_end(self, epoch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)
    time_callback = TimeHistory()
    historial = modelo.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=2, callbacks=[reduce_lr, time_callback])
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(historial.history['accuracy'], label='Train Acc', linestyle='--')
    axs[0].plot(historial.history['val_accuracy'], label='Val Acc', linestyle='--')
    axs[0].set_ylabel('Accuracy')
    axs_twin = axs[0].twinx()
    axs_twin.plot(np.cumsum(time_callback.times), label='Temps de Calcul', linestyle='-', color='red')
    axs_twin.set_ylabel('Temps (s)')
    axs_twin.spines['right'].set_position(('outward', 60))
    lines, labels = axs[0].get_legend_handles_labels()
    lines2, labels2 = axs_twin.get_legend_handles_labels()
    axs[0].legend(lines + lines2, labels + labels2)
    axs[1].plot(historial.history['loss'], label='Train Loss')
    axs[1].plot(historial.history['val_loss'], label='Val Loss')
    axs[1].set_xlabel('Époques')
    axs[1].set_ylabel('Perte')
    axs[1].legend()
    plt.tight_layout()
    plt.suptitle('Architecture: 1-50, 2-50, Out-1', fontsize=16)
    plt.show()
    return modelo, historial

df = df.dropna()
X_train, X_test, y_train, y_test = train_test(df, "RainTomorrow", 0.2)
modelo_entrenado, historial_entrenamiento = entrenar_modelo_deep_learning(X_train, y_train, X_test, y_test, epochs=200, batch_size=512, learning_rate=0.001)