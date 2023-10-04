# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 19:56:30 2023

@author: Sophie
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time

#import warnings

# preprocess
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

# optimisation
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from imblearn.over_sampling import RandomOverSampler, SMOTE

# classifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
import xgboost as xgb

# auc
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# metrics
from imblearn.metrics import classification_report_imbalanced


#warnings.filterwarnings("ignore", message="is_sparse is deprecated")

def custom_callback(params, model, X, y):
    model.fit(X, y)
    score = model.score(X, y)
    print(f"Essai des hyperparamètres : {params}")
    print(f"Score d'entraînement : {score:.4f}")


class ProjetAustralieModelisation:

    def __init__(self, data:pd.DataFrame):
        self.X=None
        self.y=None
        self.data = data.dropna()

    def _modelisation_preparation(self, cible:str, scale:bool, climat:int=None, location:str=""):
        
        # filtrage eventuel
        data=self.data
        if climat!=None:
            data = self.data[self.data.Climat==climat]
        if location!="":
            data = self.data[self.data.Location==location]
        
            
        # si pas deja lancé
        #self.preprocessing_apres_analyse()
        #self.preprocessing_basique()
        
        self.X = data.drop(columns=cible)      
        
        # supprime toutes les infos sur la meteo future
        self.X = self.X.loc[:,~self.X.columns.str.startswith("Rain_J_")]
        self.X = self.X.loc[:,~self.X.columns.str.startswith("MaxTemp_J_")]
        
        # on supprime la Location si elle est presente en str (utile uniquement pour filtrer en amont)
        if hasattr(self.X, "Location"):
            self.X = self.X.drop(columns="Location")
        
        # supprime les autres colonnes donnant trop d'indices
        if cible=="RainToday":
            self.X = self.X.drop(columns=["Rainfall"]) # si on veut predire RainToday, on supprime Rainfall, sinon c'est de la triche...
        if cible.startswith("Rain_J_"):
            self.X = self.X.drop(columns=["RainTomorrow"]) # si on veut predire RainToday, on supprime Rainfall, sinon c'est de la triche...

        #self.X = self.X.drop(columns=self.X.columns[self.X.columns.str.startswith('WindGustDir')])
        if cible.startswith("Wind"):
            self.X = self.X.drop(columns=self.X.columns[self.X.columns.str.startswith('Wind')])
        
        self.y = data[cible]
        
        # variable cible aleatoire
        ratio = len(data[data.RainTomorrow==0]) / len(data)
        #self.y = np.random.choice([0, 1], size=len(data), p=[ratio, 1-ratio])

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=66) 
        
        # normalisation
        if scale:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
        #oversample = RandomOverSampler()
        # pip install threadpoolctl==3.1.0  pour avoir SMOTE sur + de 15 colonnes
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)          
            
            
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        
    def _modelisation_matrice_confusion(self, clf):
        y_pred = clf.predict(self.X_test)
        print(pd.crosstab(self.y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))
        
        
    def modelisation(self, nom_modele:str="RandomForestClassifier", cible:str="RainTomorrow", gs:bool=True, climat:int=None, location:str=""):
               
        print (time.ctime())
        
        param_modele=None
        existe_proba=False
        clf=None
        hp=""

        print (f'\n -------\nModelisation de {cible} avec un {nom_modele}\n -------\n')
        i_temps_debut=time.time()
        
        self._modelisation_preparation(cible, True, climat, location)
        
        
        if nom_modele=='KNeighborsClassifier':
                        
            existe_proba=True
            clf = KNeighborsClassifier(n_neighbors=1)
            param_modele={ 'n_neighbors': np.arange(1,6)
                      }          
            
        elif nom_modele=='DecisionTreeClassifier':
            clf= DecisionTreeClassifier(random_state=0, max_depth=150)      

        elif nom_modele=='RandomForestClassifier':
        
            existe_proba=True
            clf = RandomForestClassifier(max_depth=8, n_estimators=100, random_state=1)            
            param_modele={ 'max_depth': range(10,21,5),
                          #'n_estimators': [5, 10, 50, 400, 800]
                          'n_estimators': [50, 100, 150]
                          }
            
        
        # boosting (chaque machine apprend à ajuster les erreurs de la precedente)
        elif nom_modele=='GradientBoostingClassifier':
            existe_proba=True
            clf=GradientBoostingClassifier(random_state=0, n_estimators=20, max_depth=4) 
            param_modele={ 'max_depth': range(9,30,1),
                          #'n_estimators': [5, 10, 50, 400, 800]
                          'n_estimators': [5, 10]
                          }

                # xgboost
        elif nom_modele=='XGBoost':
            existe_proba=True
            clf=xgb.XGBClassifier(random_state=0, learning_rate=.1, n_estimators=100, max_depth=4) 
            param_modele={ 'learning_rate': [0.1, 0.01, 0.001],
                          'n_estimators': [100, 200, 300],
                          'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                          }

        elif nom_modele=='MLPClassifier':
            
            # max 1000, hidden 1200, alpha 0.0001 : 70 71 79 90 85 (170 mn)
            existe_proba=True
            # sgd = MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(1200,), alpha=0.0001) # 74%, 8mn / 97%             
            clf = MLPClassifier(random_state=5, max_iter=150) # 74%, 8mn / 97%
        
        elif nom_modele=='DummyClassifier':
            existe_proba=True
            clf=DummyClassifier(random_state=0, strategy='stratified')
            
        else:
            print("\n -------\nNom de modèle inconnu\n -------\n")
            return
        
        # gridsv
        if param_modele!=None and gs:
            gcv = GridSearchCV(estimator=clf, 
                        param_grid=param_modele,
                        scoring='recall',
                        #scoring=score_callback,
                        #scoring='roc_auc',
                        verbose=2,
                        cv=3,
                        n_jobs=-1                        
                        )
    
            gcv.fit(self.X_train, self.y_train)
            self.gcv = gcv

            """
            for params in tqdm(param_modele):
                i_debut_fit=time.time()
                gcv.set_params(**params)
                gcv.fit(self.X_train, self.y_train)
                i_fin_fit=time.time()
                print(f"Hyper paramètres: {params} - score:{gcv.cv_results_['mean_test_score'][-1]}\n")
                print(f"Temps fit: {(i_fin_fit-i_debut_fit)/60:.2f} minutes\n\n")
            """  
            
            clf = gcv.best_estimator_
            print (gcv.best_params_)
            hp = gcv.best_params_
            self.gcv = gcv
    
        else:
            clf.fit(self.X_train, self.y_train)
        
        self.clf=clf
        
        print('Modele ', type(clf))
        
        predictions=clf.predict(self.X_train)
        i_fin_train=time.time()
        print ("Precision train: {:.2f}% - Temps train: {:.2f} minutes".format(clf.score(self.X_train, self.y_train)*100, (i_fin_train-i_temps_debut)/60))
        predictions=clf.predict(self.X_test)
        i_fin_test=time.time()
        print ("Precision test: {:.2f}% - Temps test: {:.2f} minutes".format(clf.score(self.X_test, self.y_test)*100, (i_fin_test-i_fin_train)/60))
        print (f"\nScore F1: {f1_score(self.y_test, predictions):.2f} - Accuracy: {accuracy_score(self.y_test, predictions):.2f} - Recall: {recall_score(self.y_test, predictions):.2f} - Precision: {precision_score(self.y_test, predictions):.2f}\n")
        
        
        self._modelisation_matrice_confusion(self.clf)
    
        predictions_proba=np.zeros(shape=predictions.shape)
        if existe_proba:
            predictions_proba=clf.predict_proba(self.X_test).max(axis=1)
          
        # renvoie les predictions sur le jeu complet
        #predictions=clf.predict(t_donnees_completes)
        #predictions_proba=np.zeros(shape=predictions.shape)
#        if existe_proba:
#╩            predictions_proba=clf.predict_proba(t_donnees_completes).max(axis=1)
        i_fin_completes=time.time()
        print (" Temps comp: {:.2f} minutes".format( (i_fin_completes-i_fin_test)/60))
        
        self.trace_courbe_roc(clf, nom_modele, hp, climat, location)
        
        print (time.ctime())
        
    def trace_courbe_roc(self, clf, nom_modele:str, hp:str, climat:int=None, location:str=""):
        y_test_pred_proba = clf.predict_proba(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, y_test_pred_proba[:,1])
        roc_auc = auc(fpr, tpr)
        
        diff_pr = tpr - fpr
        best_i_seuil = np.argmax(diff_pr)
        best_seuil = thresholds[best_i_seuil]
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot(fpr[best_i_seuil], tpr[best_i_seuil], 'ro', markersize=8, label=f'Seuil Optimal = {best_seuil:.2f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux Faux Positifs')
        plt.ylabel('Taux Vrais Positifs')
        
        titre_clim_loc=""        
        if climat!=None:
            titre_clim_loc="\nClimat: "+str(climat)
        if location!="":
            titre_clim_loc="\nLocation: "+location
            
            
        plt.title(f'Courbe ROC \n Modèle {nom_modele} \n {hp}{titre_clim_loc}')
        plt.legend(loc='lower right')
        plt.show()
        
        y_pred_seuil = clf.predict_proba(self.X_test)[:,1] >= best_seuil
        print (f"\nScore F1: {f1_score(self.y_test, y_pred_seuil):.2f} - Accuracy: {accuracy_score(self.y_test, y_pred_seuil):.2f} - Recall: {recall_score(self.y_test, y_pred_seuil):.2f} - Precision: {precision_score(self.y_test, y_pred_seuil):.2f}\n")
        print (f"Matrice de confusion avec seuil de {best_seuil:.2f}:")
        print(pd.crosstab(self.y_test, y_pred_seuil, rownames=['Classe réelle'], colnames=['Classe prédite']))

        print(classification_report_imbalanced(self.y_test, y_pred_seuil))

        
    def modelisation_knn(self, cible:str):
        clf_knn = KNeighborsClassifier(n_neighbors=5)              
        self._modelisation_preparation(cible, True)
        
        clf_knn.fit(self.X_train, self.y_train)
        self._modelisation_matrice_confusion(clf_knn)


    def modelisation_random_forest(self, cible:str):     
        clf_rf = RandomForestClassifier(n_jobs=-1, random_state=66 )
        self._modelisation_preparation(cible, False)
        
        clf_rf.fit(self.X_train, self.y_train)
        self._modelisation_matrice_confusion(clf_rf)
    
    
#pm = ProjetAustralieModelisation(pd.read_csv("data_basique.csv", index_col=0))
pm = ProjetAustralieModelisation(pd.read_csv("data_process3_knnim_resample_J2.csv", index_col=0))

# data_process_non_knnim => preprocession avancée (mais sans knni ni reequilibrage des classes)
# data_process2_non_knnim => idem, mais meilleure clusterisation des climats
# data_process3_knnim_resample_J2 => idem, mais knni apres drop RainTomorrow