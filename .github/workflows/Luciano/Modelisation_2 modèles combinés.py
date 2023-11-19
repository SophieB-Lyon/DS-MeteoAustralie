# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 21:10:03 2023

@author: langh
"""
# point 1.0 : estimation a été faite avec un modèle global
# point 1.0 : estimation avec un modèle global.
# point 2.0 : estimation avec un modèle individuel, un modèle par ville. Agrégation des pluies de demain des villes voisines à partir de la base de données (et non des prévisions).
# point 3.0 : comme le point 2.0 mais les valeurs de rain_tomrrow sont estimées avec un modèle ML global.



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
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from collections import Counter
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb
import re

df_base = pd.read_csv("weatherAUS.csv")
# df_base = pd.DataFrame(weatherAUScsv, columns=weatherAUScsv[0])
# df_base = df_base.drop(df_base.index[0])
# df_base.index = df_base.index - 1

#Il faut charger le fichier "df_knn_index.spydata"
df = df_knn.copy()

df_date = df_base[df_base.index.isin(df.index)]["Date"]
df_date = pd.DataFrame(df_date)
df_date.Date = pd.to_datetime(df_date["Date"]) 

ville_cluster = pd.read_csv("climats.csv", index_col="Location")
# ville_cluster = pd.DataFrame(climatscsv, columns=climatscsv[0])
# ville_cluster.set_index("Location", inplace=True)
# ville_cluster.drop(columns=ville_cluster.columns[0], inplace=True)
# ville_cluster = ville_cluster.drop(ville_cluster.index[0])

##Aajouter la colonne contenant le numéro de cluster de chaque ville
columnas_ite = df.loc[:, "Albany":"Woomera"].columns
for index in df.index :
    
    for col in columnas_ite :
                        
        if df.at[index, col]== 1:
            for ville in ville_cluster.index:
                if re.search(ville, col):
                    
                   #print(ville, pd.to_numeric(ville_cluster[ville_cluster.index == ville].values[0], errors='coerce') )
                   df.at[index, "Cluster"] = pd.to_numeric(ville_cluster[ville_cluster.index == ville].values[0], errors='coerce') 


##Pour le point 3, il est nécessaire que toutes les données de la même date soient dans le même groupe (soit Train, soit Test).
def diviser_par_date(df, cible, df_date, test_size=0.2, random_state=42):
    # Obtenez une liste unique de dates
    dates_uniques = df_date['Date'].unique()
    
    # Choisissez aléatoirement certaines dates pour l'ensemble de test
    dates_test = pd.to_datetime(pd.Series(dates_uniques).sample(frac=test_size, random_state=random_state))
    
    # Assurez-vous qu'il n'y a pas de dates en double dans dates_test
    dates_test = pd.Series(dates_test).unique()
    
    # Filtrez le DataFrame pour obtenir les lignes correspondant aux dates de test
    df_test = df[df_date['Date'].isin(dates_test)]
    
    # Les lignes restantes sont pour l'ensemble d'entraînement
    df_train = df[~df_date['Date'].isin(dates_test)]
    
    # Extraire les caractéristiques et les étiquettes
    X_train, X_test = df_train.drop(columns=[cible]), df_test.drop(columns=[cible])
    y_train, y_test = df_train[cible], df_test[cible]
    
    return X_train, X_test, y_train, y_test

def train_test(df, col_target, test_size):
    data=df.drop(col_target, axis=1)
    target = df[col_target]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, stratify=target, random_state=77)
    return X_train, X_test, y_train, y_test


def app_standard_scaler(df, col_no_scaler): ##Applique un StandardScaler à un DataFrame.
    scaler = StandardScaler()
    columnas_to_scaler = [col for col in df.columns if col not in col_no_scaler]
    df[columnas_to_scaler] = scaler.fit_transform(df[columnas_to_scaler])
    return df

def renommer_colonnes_doublons(df):
    "Renomme les colonnes en ajoutant un '2' à la fin du nom de la deuxième colonne "
    comptes_colonnes = df.columns.value_counts()
    colonnes_dupliquees = comptes_colonnes[comptes_colonnes > 1].index
    
    nouvelles_colonnes = []
    doublons = set()
    for colonne in df.columns:
        if colonne in doublons:
            nouvelle_colonne = colonne + '2'
            nouvelles_colonnes.append(nouvelle_colonne)
        else:
            nouvelles_colonnes.append(colonne)
            doublons.add(colonne)
    
    df_nouveau = df.copy()
    df_nouveau.columns = nouvelles_colonnes
    
    return df_nouveau

def trouver_meilleurs_parametres(X_train, y_train, metodo):
    #-il y a 2 XGB, un pour le global et un pour l'individuel avec moins de n_estimators
    parametres_rf = {
        'n_estimators': [25, 50, 75],
        'max_depth': [11, 15],
        'min_samples_split': [2, 3, 5],
    }
    
    parametres_xgb = {
        'n_estimators': [100, 150],
        #'eval_metric': ['auc'],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.2, 0.1, 0.05] #, 0.01
    }
    
    parametres_xgb_2 = {
        'n_estimators': [75, 125],
        #'eval_metric': ['auc'],
        'max_depth': [2, 4, 6],
        'learning_rate': [0.2, 0.1, 0.05] #, 0.01
    }
    
    if metodo == 1:
        # Utilizar Random Forest
        clf = RandomForestClassifier()
        parametres = parametres_rf
    elif metodo == 2:
        # Utilizar XGBoost
        clf = XGBClassifier()
        parametres = parametres_xgb
    
    elif metodo == 3:
        # Utilizar XGBoost
        #ratio = sum(y_train == 0) / sum(y_train == 1)
        clf = XGBClassifier() #scale_pos_weight=ratio
        parametres = parametres_xgb_2
        
    # Initialiser a recherche en grille avec validation croisée (cross-validation)
    recherche_en_grille = GridSearchCV(clf, parametres, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    # Trouver les meilleurs hyperparamètres
    recherche_en_grille.fit(X_train, y_train)
    best_params = recherche_en_grille.best_params_
    
    return recherche_en_grille.best_estimator_, best_params


## Test de differetns Modeles
modele_global = 2
modele_indiv = 3  # 1 for RF et 2 pour KGBOOST
## Division de X_train pour utiliser le meem avec les differents modeles

df = df.dropna()
#df_base = df_base.drop(df_base.loc[:, df_base['Location'] == 'Adelaine'].columns, axis=1)
df = renommer_colonnes_doublons(df)
X_train, X_test, y_train, y_test = train_test(df, "RainTomorrow", 0.3)



### 1.1 Modele Global (un seule modele pour tous les villes sans cluster)
X_train_sc = X_train.drop(columns="Cluster")
X_test_sc = X_test.drop(columns="Cluster")
clf, best_params= trouver_meilleurs_parametres(X_train_sc, y_train, modele_global)
y_test_pred = clf.predict(X_test_sc)
y_test_prob = clf.predict_proba(X_test_sc)[:, 1]
print("1.1:", best_params)
print("train", clf.score(X_train_sc, y_train))
print("test", clf.score(X_test_sc, y_test))


score_global = []
f1_global = []
villes = []
score_base = []
f1_base = []
roc_global = []

y_test_pred_df = pd.DataFrame(y_test_pred)
y_test_pred_df["Index"] = X_test.index.tolist()
y_test_prob_df = pd.DataFrame(y_test_prob)
y_test_prob_df["Index"] = X_test.index.tolist()


for ville in df_base.Location.unique():
    try:
        x_test_aux = X_test_sc[X_test_sc[ville] > 0]
        y_test_aux = y_test.loc[x_test_aux.index]
        y_test_pred_aux = y_test_pred_df[y_test_pred_df["Index"].isin(x_test_aux.index)][0]
        y_test_prob_aux = y_test_prob_df[y_test_prob_df["Index"].isin(x_test_aux.index)][0]

        #print("el score pour", ville, "est :", clf.score(x_test_aux, y_test_aux))
        #print("el F1 pour", ville, "est :", f1_score(y_test_aux, y_test_pred_aux))
        villes.append(ville)
        score_global.append(clf.score(x_test_aux, y_test_aux))
        f1_global.append(f1_score(y_test_aux, y_test_pred_aux))
        roc_global.append(roc_auc_score(y_test_aux, y_test_prob_aux))
        
       
        y_test_pred_aux = pd.Series(np.zeros(len(y_test_aux)))
        matrix = confusion_matrix(y_test_aux, y_test_pred_aux)
        SCORE = (matrix[0][0]+matrix[1][1])/(matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1])
        score_base.append(SCORE)
               
        
    except KeyError:
        print("error avec", ville)
        

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_global.append(np.mean([i for i in score_global if i !=0]))
score_base.append(np.mean([i for i in score_base if i !=0]))
f1_global.append(np.mean([i for i in f1_global if i !=0]))
roc_global.append(np.mean([i for i in roc_global if i !=0]))

## 1.2 Graphes Modele Global
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
plt.ylabel('SCORE')
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi, f1_global, width=space, color='r', label='ML Global')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.4,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('F1')
plt.show()
     
#Afficher Curve ROC
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = roc_auc_score(y_test, y_test_prob)

# Graf
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()



### 2.0 Ajouter predccion des autres villes (pas pour la même ville) en tant que features

## 2.1  Test théorique : j'ajoute le raintomorrw de la base de données pour voir le maximum théorique que l'on peut atteindre.
#df_ext : une colonne par ville est ajoutée avec vos données raintomorrow. Les données raintomorrow et null sont remplies de zéros. 
df_ext = df.copy()
for index, row in df_ext.iterrows():
        
    # Obtenir la date correspondante dans df_date pour chaque ligne de df_ext
    fecha_actual = df_date.loc[index, 'Date']  # Reemplaza 'tu_columna_de_fecha' con el nombre real de la columna
    
    # Obtenir tous les index de df_date qui ont la même date
    indices_misma_fecha = df_date[df_date['Date'] == fecha_actual].index.tolist()
    indices_misma_fecha = [i for i in indices_misma_fecha if i in df_ext.index]
    
    # Obtenir la prédiction pour ces indices dans y_train
    valores_RainTomorrow = df_ext.RainTomorrow.loc[indices_misma_fecha].drop(index=index) #"drop(index=index)" C'est pour supprimer les données pour la même ville. Par exemple pour Sydney j'ajoute le raintomorrow pour toutes les villes sauf Sydney. 
    
    # Agregar los valores como nuevas columnas a la fila actual de X_train
    for i, valor in enumerate(valores_RainTomorrow):
        ville = df_base.at[valores_RainTomorrow.index[i], "Location"]
        #ville = (df.loc[valores_RainTomorrow.index[i], 'Albany':'Woomera'] == 1).idxmax()
        columna_nueva = f'Rain_{ville}'
        df_ext.loc[index, columna_nueva] = valor
        #print(ville, valores_RainTomorrow.index[i])

df_ext.fillna(0, inplace=True)


X_train_ext, X_test_ext, y_train_ext, y_tes_ext = diviser_par_date(df_ext, "RainTomorrow", df_date, 0.3, 77)
## Avec donnes reel
score_indiv_MaxTheo = []
f1_indiv_MaxTheo = []
roc_indiv_MaxTheo = []
villes = []
for ville in df_base.Location.unique():
    try:
        x_train_aux = X_train_ext[X_train_ext[ville] > 0]
        y_train_aux = y_train_ext.loc[x_train_aux.index]
        x_test_aux = X_test_ext[X_test_ext[ville] > 0]
        y_test_aux = y_tes_ext.loc[x_test_aux.index]
                        
        clf, best_params = trouver_meilleurs_parametres(x_train_aux, y_train_aux, modele_indiv)
        y_test_pred_aux = clf.predict(x_test_aux)
        y_test_prob_aux = clf.predict_proba(x_test_aux)[:, 1]
        score_indiv_MaxTheo.append(clf.score(x_test_aux, y_test_aux))
        f1_indiv_MaxTheo.append(f1_score(y_test_aux, y_test_pred_aux))
        roc_indiv_MaxTheo.append(roc_auc_score(y_test_aux, y_test_prob_aux))
        villes.append(ville)
        print("3.1:", best_params)
        print("test", clf.score(x_test_aux, y_test_aux))
        print("train", clf.score(x_train_aux, y_train_aux))
        
        #Graf importances de features 
        noms_variables = list(x_train_aux.columns)
        importances = clf.feature_importances_
        indices = importances.argsort()[::-1]
        noms_variables_tries = [noms_variables[i] for i in indices]

        # Ggraphique à barres
        plt.figure(figsize=(10, 25))  # Ajuster la taille selon les besoins
        plt.barh(noms_variables_tries, importances[indices])
        plt.xlabel('Importance')
        plt.title('Importance des Variables'f'{ ville}')
        plt.gca().invert_yaxis()  # Inverser l'ordre des variables
        plt.show()
               
        
    except KeyError:
        print("error avec", ville)
        

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_indiv_MaxTheo.append(np.mean([i for i in score_indiv_MaxTheo if i !=0]))
f1_indiv_MaxTheo.append(np.mean([i for i in f1_indiv_MaxTheo if i !=0]))
roc_indiv_MaxTheo.append(np.mean([i for i in roc_indiv_MaxTheo if i !=0]))

#Afficher le score par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.26
plt.bar(indi, score_global, width=space, color='r', label='ML Global')
#plt.bar(indi+space, score_global, width=space, color='b', label='CLUSTER global')
plt.bar(indi+space, score_indiv_MaxTheo, width=space, color='g', label='ML Ind_MaxTheo')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.6,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('SCORE')
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi, f1_global, width=space, color='r', label='ML Global')
#plt.bar(indi+space, f1_global, width=space, color='b', label='CLUSTER global')
plt.bar(indi+space*2, f1_indiv_MaxTheo, width=space, color='g', label='ML Ind_MaxTheo')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.3,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('F1')
plt.show()

#Afficher le ROC par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi, roc_global, width=space, color='r', label='ML Global')
plt.bar(indi+space, roc_indiv_MaxTheo, width=space, color='g', label='ML Ind_MaxTheo')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.75,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('ROC')
plt.show()


#### 3.0 -   2 modèles en série (un global et un individuel)- Avec et sans la prédiction de la même ville
### 3.1 Premier ML global 
X_train, X_test, y_train, y_test = diviser_par_date(df, "RainTomorrow", df_date, 0.3, 77)
clf_ext, best_params = trouver_meilleurs_parametres(X_train, y_train, modele_global)
y_test_pred = clf_ext.predict(X_test)
y_test_pred = pd.DataFrame(y_test_pred, index=X_test.index, columns=['Predicciones'])
y_train_pred = clf_ext.predict(X_train)
y_train_pred = pd.DataFrame(y_train_pred, index=X_train.index, columns=['Predicciones'])



### 3.2 Deuxième modèle individuel par ville - SANS la prédiction de la ville elle-même

##Construir X_test_ext_pred a partir de las predicciones de premier Modele
#♣Ajouter colonnes avec RainTOmorrow des autres villes -  sans ville elle meme
X_test_ext_pred = X_test.copy() 

for index, row in X_test_ext_pred.iterrows():
    
    #index = 129837
    # obtenir la date correspondante dans df_date
    fecha_actual = df_date.loc[index, 'Date']  
    
    # Obtenir tous les index de df_date qui ont la même date
    indices_misma_fecha = df_date[df_date['Date'] == fecha_actual].index.tolist()
    indices_misma_fecha = [i for i in indices_misma_fecha if i in X_test.index]
    
    # Obtenir les valeurs correspondantes dans y_train
    valores_RainTomorrow = y_test_pred.Predicciones.loc[indices_misma_fecha].drop(index=index)
    
    # Agregar los valores como nuevas columnas a la fila actual de X_train
    for i, valor in enumerate(valores_RainTomorrow):
        ville = df_base.at[valores_RainTomorrow.index[i], "Location"]
        #print(ville)
        columna_nueva = f'Rain_{ville}'
        X_test_ext_pred.loc[index, columna_nueva] = valor
X_test_ext_pred.fillna(0, inplace=True)

## Calcules Deuxieme ML
score_indiv = []
f1_indiv = []
roc_indiv = []
villes = []
for ville in df_base.Location.unique():
    try:
        x_train_aux = X_train_ext[X_train[ville] > 0]
        y_train_aux = y_train.loc[x_train_aux.index]
        x_test_aux = X_test_ext_pred[X_test_ext[ville] > 0]
        y_test_aux = y_test.loc[x_test_aux.index]
                        
        clf, best_params = trouver_meilleurs_parametres(x_train_aux, y_train_aux, modele_indiv)
        y_test_pred_aux = clf.predict(x_test_aux)
        y_test_prob_aux = clf.predict_proba(x_test_aux)[:, 1]
        score_indiv.append(clf.score(x_test_aux, y_test_aux))
        f1_indiv.append(f1_score(y_test_aux, y_test_pred_aux))
        roc_indiv.append (roc_auc_score(y_test_aux, y_test_prob_aux))
        villes.append(ville)
        print("3.2:", best_params, ville)
        print("test", clf.score(x_test_aux, y_test_aux))
        print("train", clf.score(x_train_aux, y_train_aux))
        
        #Graf importances de features 
        noms_variables = list(x_train_aux.columns)
        importances = clf.feature_importances_
        indices = importances.argsort()[::-1]
        noms_variables_tries = [noms_variables[i] for i in indices]

        # Ggraphique à barres
        plt.figure(figsize=(10, 25))  # Ajuster la taille selon les besoins
        plt.barh(noms_variables_tries, importances[indices])
        plt.xlabel('Importance')
        plt.title('Importance des Variables'f'{ ville}')
        plt.gca().invert_yaxis()  # Inverser l'ordre des variables
        plt.show()
               
        
    except KeyError:
        print("error avec", ville)
        

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_indiv.append(np.mean([i for i in score_indiv if i !=0]))
f1_indiv.append(np.mean([i for i in f1_indiv if i !=0]))
roc_indiv.append(np.mean([i for i in roc_indiv if i !=0]))

#Afficher le score par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.26
plt.bar(indi, score_global, width=space, color='r', label='ML Global')
#plt.bar(indi+space, score_global, width=space, color='b', label='CLUSTER global')
plt.bar(indi+space, score_indiv, width=space, color='g', label='ML Global+Ind')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.7,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('SCORE')
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi, f1_global, width=space, color='r', label='ML Global')
#plt.bar(indi+space, f1_global, width=space, color='b', label='CLUSTER global')
plt.bar(indi+space, f1_indiv, width=space, color='g', label='ML Global+Ind')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.4,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('F1')
plt.show()

#Afficher le ROC par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi, roc_global, width=space, color='r', label='ML Global')
plt.bar(indi+space, roc_indiv, width=space, color='g', label='ML Global+Ind')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.6,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('ROC')
plt.show()




### 3.3 Deuxième modèle individuel par ville - AVEC la prédiction de la ville elle-même

# le premier modèle global a été réalisé en 3.1

##Construir X_train_ext_pred avec les données de la base de données pour les villes voisines +
## + la prévision pour la même ville.
## (Pour la même ville, je ne peux pas utiliser les données de la base de données car il s'agit de la variable à prédire.)


# Adapter X_train_ext avec predicctions(incluir prediction meme vile) :
X_train_ext_pred = X_train_ext.copy()
for index, row in X_train_ext_pred.iterrows():
    
    # obtenir la date correspondante dans df_date
    fecha_actual = df_date.loc[index, 'Date']  
    
    # # Obtenir tous les index de df_date qui ont la même date
    # indices_misma_fecha = df_date[df_date['Date'] == fecha_actual].index.tolist()
    # indices_misma_fecha = [i for i in indices_misma_fecha if i in df_ext.index]
    
    # Obtenir le valeur correspondantes dans y_train pour la ville elle meme
    valor_RainTomorrow = y_train_pred.Predicciones.loc[index]
    
    # Agregar los valores como nuevas columnas a la fila actual de X_train
    ville = df_base.at[index, "Location"]
    columna_nueva = f'Rain_{ville}'
    X_train_ext_pred.loc[index, columna_nueva] = valor_RainTomorrow



##Construir X_test_ext_pred a partir de las predicciones de premier Modele
#♣Ajouter colonnes avec RainTOmorrow de chaque ville
X_test_ext_pred_mv = X_test.copy()
X_test_ext_pred_mv = X_test_ext_pred.reindex(columns=X_train_ext_pred.columns, fill_value=None)

for index, row in X_test_ext_pred.iterrows():
    
    #index = 129837
    # obtenir la date correspondante dans df_date
    fecha_actual = df_date.loc[index, 'Date']  
    
    # Obtenir tous les index de df_date qui ont la même date
    indices_misma_fecha = df_date[df_date['Date'] == fecha_actual].index.tolist()
    indices_misma_fecha = [i for i in indices_misma_fecha if i in X_test.index]
    
    # Obtenir les valeurs correspondantes dans y_train
    valores_RainTomorrow = y_test_pred.Predicciones.loc[indices_misma_fecha]
    
    # Agregar los valores como nuevas columnas a la fila actual de X_train
    for i, valor in enumerate(valores_RainTomorrow):
        ville = df_base.at[valores_RainTomorrow.index[i], "Location"]
        #print(ville)
        columna_nueva = f'Rain_{ville}'
        X_test_ext_pred_mv.loc[index, columna_nueva] = valor
X_test_ext_pred_mv.fillna(0, inplace=True)

## Calcules Deuxieme ML
score_indiv_mv = []
f1_indiv_mv = []
roc_indiv_mv = []
villes = []
for ville in df_base.Location.unique():
    try:
        x_train_aux = X_train_ext_pred[X_train[ville] > 0]
        y_train_aux = y_train.loc[x_train_aux.index]
        x_test_aux = X_test_ext_pred_mv[X_test_ext[ville] > 0]
        y_test_aux = y_test.loc[x_test_aux.index]
        
       
                        
        clf, best_params = trouver_meilleurs_parametres(x_train_aux, y_train_aux, modele_indiv)
        y_test_pred_aux = clf.predict(x_test_aux)
        y_test_prob_aux = clf.predict_proba(x_test_aux)[:, 1]
        score_indiv_mv.append(clf.score(x_test_aux, y_test_aux))
        f1_indiv_mv.append(f1_score(y_test_aux, y_test_pred_aux))
        roc_indiv_mv.append (roc_auc_score(y_test_aux, y_test_prob_aux))
        villes.append(ville)
        print("3.3:", best_params, ville)
        print("test", clf.score(x_test_aux, y_test_aux))
        print("train", clf.score(x_train_aux, y_train_aux))
        #print(df_base.loc[df_base.index.isin(x_train_aux.index), 'Location'])
        
        #Graf importances de features 
        noms_variables = list(x_train_aux.columns)
        importances = clf.feature_importances_
        indices = importances.argsort()[::-1]
        noms_variables_tries = [noms_variables[i] for i in indices]

        # Ggraphique à barres
        plt.figure(figsize=(10, 25))  # Ajuster la taille selon les besoins
        plt.barh(noms_variables_tries, importances[indices])
        plt.xlabel('Importance')
        plt.title('Importance des Variables'f'{ ville}')
        plt.gca().invert_yaxis()  # Inverser l'ordre des variables
        plt.show()
               
        
    except KeyError:
        print("error avec", ville)
        

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_indiv_mv.append(np.mean([i for i in score_indiv_mv if i !=0]))
f1_indiv_mv.append(np.mean([i for i in f1_indiv_mv if i !=0]))
roc_indiv_mv.append(np.mean([i for i in roc_indiv_mv if i !=0]))

#Afficher le score par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.26
plt.bar(indi, score_global, width=space, color='r', label='ML Global')
#plt.bar(indi+space, score_global, width=space, color='b', label='CLUSTER global')
plt.bar(indi+space, score_indiv_mv, width=space, color='g', label='ML Global+Ind_MV')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.7,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('SCORE')
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi, f1_global, width=space, color='r', label='ML Global')
#plt.bar(indi+space, f1_global, width=space, color='b', label='CLUSTER global')
plt.bar(indi+space, f1_indiv_mv, width=space, color='g', label='ML Global+Ind_MV')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.4,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('F1')
plt.show()

#Afficher le ROC par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi, roc_global, width=space, color='r', label='ML Global')
plt.bar(indi+space, roc_indiv_mv, width=space, color='g', label='ML Global+Ind_MV')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.6,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('ROC')
plt.show()
