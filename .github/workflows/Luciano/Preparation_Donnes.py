# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:45:39 2023

@author: langh
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
from imblearn.under_sampling import ClusterCentroids


def transform_varia_cualit(df, columns_to_trans):
    df_trans = pd.get_dummies(df, columns=columns_to_trans, drop_first=True, prefix='', prefix_sep='')
    return df_trans

def select_numeric(dataframe):
    # Selecct variables numeriques ## to improve later
    numeric_columns = dataframe.select_dtypes(include=[pd.np.number])
    return numeric_columns

def knn_for_NNA(df, n_neighbors):
    column_names = df.columns
    knn_imputer = KNNImputer(n_neighbors= n_neighbors)
    df_knn = knn_imputer.fit_transform(df)
    df_knn = pd.DataFrame(df_knn, columns=column_names)
    return df_knn

def app_standard_scaler(df, col_no_scaler): ##Applique un StandardScaler à un DataFrame.
    scaler = StandardScaler()
    columnas_to_scaler = [col for col in df.columns if col not in col_no_scaler]
    df[columnas_to_scaler] = scaler.fit_transform(df[columnas_to_scaler])
    return df

# def select_numeric(dataframe):
#     # Selecct variables numeriques ## to improve later
#     numeric_columns = dataframe.select_dtypes(include=[pd.np.number])
#     return numeric_columns


df = pd.read_csv("weatherAUS")


df.RainToday = df['RainToday'].replace({'Yes': 1, 'No': 0})
df.RainTomorrow = df['RainTomorrow'].replace({'Yes': 1, 'No': 0})
#df = df.reset_index(drop=True)
df.dropna(subset=["RainTomorrow"], inplace=True)   ##SUPRIMER rows RainTomorrow nulos
print(df.dtypes)



df_aux1 = df.select_dtypes(include=["number"]) #Séparer les valeurs numériques pour Scaler
df_aux2 = df.select_dtypes(exclude=['number']) #Séparer les non-valeurs numériques pour Get_dummies
df_aux1 = app_standard_scaler(df_aux1, ["RainToday", "RainTomorrow"])
df_aux2 = transform_varia_cualit(df_aux2, ["Location", "WindDir3pm", "WindDir9am"])      
df_r = pd.merge(df_aux1, df_aux2, left_index=True, right_index=True)
df_r = select_numeric(df_r)       #Suprimmer column non-numeric -> la DATE est suprimme

df_knn = knn_for_NNA(df_r, 4)
df_knn = df_knn.set_index(df_r.index)

ruta = 'C:/Users/langh/OneDrive/Desktop/adm'
df_knn.to_excel(ruta, engine='openpyxl')