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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import ClusterCentroids

def reductor_df(dataframe, colum_date, fraction):  #Reduir taille de Dataframe
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

def train_test(df, col_target, test_size):
    data=df.drop(col_target, axis=1)
    target = df[col_target]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, stratify=target, random_state=77)
    return X_train, X_test, y_train, y_test


def remove_cities_with_NNA(df, num_citys):
    missing_percentages = df.groupby("Location").apply(lambda x: x.isnull().mean().mean())
    cities_to_remove = missing_percentages.sort_values(ascending=False).index[:num_citys]
    df_filtered = df.loc[~df["Location"].isin(cities_to_remove)]
    return df_filtered

def app_standard_scaler(df, col_no_scaler): ##Applique un StandardScaler à un DataFrame.
    scaler = StandardScaler()
    columnas_to_scaler = [col for col in df.columns if col not in col_no_scaler]
    df[columnas_to_scaler] = scaler.fit_transform(df[columnas_to_scaler])
    return df
    

# def app_standard_scaler(df): ##Applique un StandardScaler à un DataFrame.
#     scaler = StandardScaler()
#     donees = scaler.fit_transform(df)
#     df_scaled = pd.DataFrame(donees, columns=df.columns, index=df.index)
#     return df_scaled

df = pd.read_csv("weatherAUS.csv")
df.RainToday = df['RainToday'].replace({'Yes': 1, 'No': 0})
df.RainTomorrow = df['RainTomorrow'].replace({'Yes': 1, 'No': 0})
df = df.reset_index(drop=True)
df.dropna(subset=["RainTomorrow"], inplace=True)   ##SUPRIMER rows RainTomorrow nulos

df_r = reductor_df(df, "Date", 1  )   #Reduir taille de Dataframe
df_aux1 = df_r.select_dtypes(include=["number"])
df_aux2 = df_r.select_dtypes(exclude=['number'])
df_aux1 = app_standard_scaler(df_aux1, ["RainToday", "RainTomorrow"])
df_aux2 = transform_varia_cualit(df_aux2, ["Location", "WindDir3pm", "WindDir9am"])      
df_r = pd.merge(df_aux1, df_aux2, left_index=True, right_index=True)
df_r = select_numeric(df_r)       #Suprimmer column non-numeric -> la DATE est suprimme
df_knn = knn_for_NNA(df_r, 4)







X_train, X_test, y_train, y_test = train_test(df_knn, "RainTomorrow", 0.3)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
#result = pd.crosstab(y_test, y_test_pred)
print("el score ets=", clf.score(X_test, y_test))
print("el F1 est=", f1_score(y_test, y_test_pred))


## Je veux voir le SCORE et f1 par ville avec un modèle global (un seul modèle pour toutes les villes), et ensuite comparer avec une méthode ML par ville.


score_global = []
f1_global = []
villes = []

score_base = []
f1_base = []

y_test_pred_df = pd.DataFrame(y_test_pred)
y_test_pred_df["Index"] = X_test.index.tolist()


for ville in df.Location.unique():
    try:
        x_test_aux = X_test[X_test[ville] > 0]
        y_test_aux = y_test.loc[x_test_aux.index]
        y_test_pred_aux = y_test_pred_df[y_test_pred_df["Index"].isin(x_test_aux.index)][0]

        #print("el score pour", ville, "est :", clf.score(x_test_aux, y_test_aux))
        #print("el F1 pour", ville, "est :", f1_score(y_test_aux, y_test_pred_aux))
        villes.append(ville)
        score_global.append(clf.score(x_test_aux, y_test_aux))
        f1_global.append(f1_score(y_test_aux, y_test_pred_aux))
        
       
        y_test_pred_aux = pd.Series(np.zeros(len(y_test_aux)))
        matrix = confusion_matrix(y_test_aux, y_test_pred_aux)
        SCORE = (matrix[0][0]+matrix[1][1])/(matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1])
        score_base.append(SCORE)
               
        
    except KeyError:
        print("error avec", ville)
        villes.append(ville)
        score_global.append(0)
        f1_global.append(0)
        score_base.append(0)

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_global.append(np.mean([i for i in score_global if i !=0]))
score_base.append(np.mean([i for i in score_base if i !=0]))
f1_global.append(np.mean([i for i in f1_global if i !=0]))

#Afficher le score par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.35
plt.bar(indi, score_global, width=space, color='r', label='ML Global')
plt.bar(indi+space, score_base, width=space, color='b', label='Pred Base')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.6,1)
plt.grid(True, axis='y', alpha=0.8)
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi+space, f1_global, width=space, color='r', label='ML Global')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.4,1)
plt.grid(True, axis='y', alpha=0.8)
plt.show()
     


## Calculer le Score et le F1 pour cahque ville avec modèle individuel (un modèle par ville)
score_indiv = []
f1_indiv = []
villes = []
for ville in df.Location.unique():
    try:
        x_train_aux = X_train[X_train[ville] > 0]
        y_train_aux = y_train.loc[x_train_aux.index]
        x_test_aux = X_test[X_test[ville] > 0]
        y_test_aux = y_test.loc[x_test_aux.index]
                        
        clf = RandomForestClassifier()
        clf.fit(x_train_aux, y_train_aux)
        y_test_pred_aux = clf.predict(x_test_aux)
        score_indiv.append(clf.score(x_test_aux, y_test_aux))
        f1_indiv.append(f1_score(y_test_aux, y_test_pred_aux))
        villes.append(ville)
               
        
    except KeyError:
        score_indiv.append(0)
        f1_indiv.append(0)
        villes.append(ville)

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_indiv.append(np.mean([i for i in score_indiv if i !=0]))
f1_indiv.append(np.mean([i for i in f1_indiv if i !=0]))

#Afficher le score par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.26
plt.bar(indi, score_indiv, width=space, color='g', label='ML Individuel')
plt.bar(indi+space, score_global, width=space, color='r', label='ML Global')
plt.bar(indi+space*2, score_base, width=space, color='b', label='Pred Bas')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.6,1)
plt.grid(True, axis='y', alpha=0.8)
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.35
plt.bar(indi, f1_indiv, width=space, color='g', label='ML Individuel')
plt.bar(indi+space, f1_global, width=space, color='r', label='ML Global')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.3,1)
plt.grid(True, axis='y', alpha=0.8)
plt.show()



## Calculer le Score et le F1 pour cahque ville avec modèle individuel + ajustement du déséquilibre
score_indiv_under = []
f1_indiv_under = []
villes = []
for ville in df.Location.unique():
    try:
              
        x_train_aux = X_train[X_train[ville] > 0]
        y_train_aux = y_train.loc[x_train_aux.index]
        x_test_aux = X_test[X_test[ville] > 0]
        y_test_aux = y_test.loc[x_test_aux.index]
        
        # #undersampling
        # cc = ClusterCentroids()
        # x_train_aux_cc,  y_train_aux_cc = cc.fit_resample(x_train_aux, y_train_aux)
               
        clf = RandomForestClassifier(class_weight="balanced")
        clf.fit(x_train_aux, y_train_aux)
        y_test_pred_aux = clf.predict(x_test_aux)
        score_indiv_under.append(clf.score(x_test_aux, y_test_aux))
        f1_indiv_under.append(f1_score(y_test_aux, y_test_pred_aux))
        villes.append(ville)
               
        
    except KeyError:
        score_indiv_under.append(0)
        f1_indiv_under.append(0)
        villes.append(ville)

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_indiv_under.append(np.mean([i for i in score_indiv_under if i !=0]))
f1_indiv_under.append(np.mean([i for i in f1_indiv_under if i !=0]))

#Afficher le score par ville
plt.figure(figsize=(30, 10))
indi = np.arange(len(villes))
space = 0.2
plt.bar(indi, score_indiv_under, width=space, color='black', label='ML desequilibre-ind-')
plt.bar(indi+space, score_indiv, width=space, color='g', label='ML Individuel')
plt.bar(indi+space*2, score_global, width=space, color='r', label='ML Global')
plt.bar(indi+space*3, score_base, width=space, color='b', label='Pred Base')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.6,1)
plt.grid(True, axis='y', alpha=0.8)
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(30, 10))
indi = np.arange(len(villes))
space = 0.25
plt.bar(indi, f1_indiv_under, width=space, color='black', label='ML desequilibre-ind-')
plt.bar(indi+space, f1_indiv, width=space, color='g', label='ML Individuel')
plt.bar(indi+space*2, f1_global, width=space, color='r', label='ML Global')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.4,1)
plt.grid(True, axis='y', alpha=0.8)
plt.show()





## Calculer si on ajoute les donnes de la veille

df_dup = df_knn
aux = df_dup.shift(1)
df_veille = pd.merge(df_knn, aux, left_index=True, right_index=True, how='right', suffixes=("","-1"))
df_veille.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test(df_veille, "RainTomorrow", 0.2)

score_veille = []
f1_veille = []
villes = []
for ville in df.Location.unique():
    try:
        x_train_aux = X_train[X_train[ville] > 0]
        y_train_aux = y_train.loc[x_train_aux.index]
        x_test_aux = X_test[X_test[ville] > 0]
        y_test_aux = y_test.loc[x_test_aux.index]
                        
        clf = RandomForestClassifier()
        clf.fit(x_train_aux, y_train_aux)
        y_test_pred_aux = clf.predict(x_test_aux)
        score_veille.append(clf.score(x_test_aux, y_test_aux))
        f1_veille.append(f1_score(y_test_aux, y_test_pred_aux))
        villes.append(ville)
               
        
    except KeyError:
        score_veille.append(0)
        f1_veille.append(0)
        villes.append(ville)

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_veille.append(np.mean([i for i in score_veille if i !=0]))
f1_veille.append(np.mean([i for i in f1_veille if i !=0]))

#Afficher le score par ville
plt.figure(figsize=(30, 10))
indi = np.arange(len(villes))
space = 0.21
plt.bar(indi, score_veille, width=space, color='black', label='ML Veille-ind-')
plt.bar(indi+space, score_indiv, width=space, color='g', label='ML Individuel')
plt.bar(indi+space*2, score_global, width=space, color='r', label='ML Global')
plt.bar(indi+space*3, score_base, width=space, color='b', label='Pred Base')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.6,1)
plt.grid(True, axis='y', alpha=0.8)
plt.show()



#Afficher le f1 par ville
plt.figure(figsize=(30, 10))
indi = np.arange(len(villes))
space = 0.25
plt.bar(indi, f1_veille, width=space, color='black', label='ML Veille_ind_')
plt.bar(indi+space, f1_indiv, width=space, color='g', label='ML Individuel')
plt.bar(indi+space*2, f1_global, width=space, color='r', label='ML Global')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.4,1)
plt.grid(True, axis='y', alpha=0.8)
plt.show()




## Je veux voir le SCORE et f1 par ville avec un modèle global avec les donnes de la veille

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)

score_global_veille = []
f1_global_veille = []
villes = []

score_base = []
f1_base = []

y_test_pred_df = pd.DataFrame(y_test_pred)
y_test_pred_df["Index"] = X_test.index.tolist()

for ville in df.Location.unique():
    try:
        x_test_aux = X_test[X_test[ville] > 0]
        y_test_aux = y_test.loc[x_test_aux.index]
        y_test_pred_aux = y_test_pred_df[y_test_pred_df["Index"].isin(x_test_aux.index)][0]

        #print("el score pour", ville, "est :", clf.score(x_test_aux, y_test_aux))
        #print("el F1 pour", ville, "est :", f1_score(y_test_aux, y_test_pred_aux))
        villes.append(ville)
        score_global.append(clf.score(x_test_aux, y_test_aux))
        f1_global.append(f1_score(y_test_aux, y_test_pred_aux))
        
       
        y_test_pred_aux = pd.Series(np.zeros(len(y_test_aux)))
        matrix = confusion_matrix(y_test_aux, y_test_pred_aux)
        SCORE = (matrix[0][0]+matrix[1][1])/(matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1])
        score_base.append(SCORE)
               
        
    except KeyError:
        print("error avec", ville)
        villes.append(ville)
        score_global.append(0)
        f1_global.append(0)
        score_base.append(0)

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_global.append(np.mean([i for i in score_global if i !=0]))
score_base.append(np.mean([i for i in score_base if i !=0]))
f1_global.append(np.mean([i for i in f1_global if i !=0]))

#Afficher le score par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.35
plt.bar(indi, score_global, width=space, color='r', label='ML Global')
plt.bar(indi+space, score_base, width=space, color='b', label='Pred Base')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.6,1)
plt.grid(True, axis='y', alpha=0.8)
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi+space, f1_global, width=space, color='r', label='ML Global')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.4,1)
plt.grid(True, axis='y', alpha=0.8)
plt.show()