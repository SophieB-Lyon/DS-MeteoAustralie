# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 14:14:29 2023

@author: langh
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



df = pd.read_csv("weatherAUS.csv")
df.RainToday = df['RainToday'].replace({'Yes': 1, 'No': 0})
df.RainTomorrow = df['RainTomorrow'].replace({'Yes': 1, 'No': 0})
df = df.reset_index(drop=True)
df.dropna(subset=["RainTomorrow"], inplace=True)   ##SUPRIMER rows RainTomorrow nulos



def reductor_df(dataframe, colum_date, fraction):
    # Converter to date format
    dataframe[colum_date] = pd.to_datetime(dataframe[colum_date])
    # Ordenar
    dataframe = dataframe.sort_values(by=colum_date)
    # Calcul qty donnes
    index_cut = int(len(dataframe) * fraction)
    # Select les premieres donnes
    filtered_dataframe = dataframe.head(index_cut)
    return filtered_dataframe

def transform_varia_cualit(df, columns_to_trans):
    df_trans = pd.get_dummies(df, columns=columns_to_trans, drop_first=True)
    return df_trans

def select_numeric(dataframe):
    # Selecct variables numeriques ## to improve later
    numeric_columns = dataframe.select_dtypes(include=[pd.np.number])
    return numeric_columns

def knn_for_NNA(df, n_neighbors):
    column_names = df.columns
    knn_imputer = KNNImputer(n_neighbors=1)
    df_knn = knn_imputer.fit_transform(df)
    df_knn = pd.DataFrame(df_knn, columns=column_names)
    return df_knn

def train_test(df, col_target, test_size):
    data=df.drop(col_target, axis=1)
    target = df[col_target]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=56)
    return X_train, X_test, y_train, y_test


def remove_cities_with_NNA(df, num_citys):
    missing_percentages = df.groupby("Location").apply(lambda x: x.isnull().mean().mean())
    cities_to_remove = missing_percentages.sort_values(ascending=False).index[:num_citys]
    df_filtered = df.loc[~df["Location"].isin(cities_to_remove)]
    return df_filtered



df = remove_cities_with_NNA(df, 4)
df_r = reductor_df(df, "Date", 0.07)   #Reduir taille de Dataframe
df_r = transform_varia_cualit(df_r, ["Location", "WindDir3pm", "WindDir9am"])
df_r = select_numeric(df_r)            #Suprimmer column non-numeric
df_knn = knn_for_NNA(df_r, 2)
X_train, X_test, y_train, y_test = train_test(df_knn, "RainTomorrow", 0.25)


clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
result = pd.crosstab(y_test, y_test_pred)
print(result)
print("El SCORE est :", clf.score(X_test, y_test))


