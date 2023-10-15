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

import warnings

# preprocess
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

# optimisation
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from imblearn.over_sampling import RandomOverSampler, SMOTE
# auc
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# metrics
from sklearn.metrics import make_scorer
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import mean_squared_error, mean_absolute_error

# classifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
import xgboost as xgb


# timeseries
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm       
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# pour couleurs
import plotly.express as px
import plotly.colors as pc

# divers
from tqdm import tqdm

#warnings.filterwarnings("ignore", message="is_sparse is deprecated")
warnings.filterwarnings("ignore", category=UserWarning)

def custom_callback(params, model, X, y):
    model.fit(X, y)
    score = model.score(X, y)
    print(f"Essai des hyperparamètres : {params}")
    print(f"Score d'entraînement : {score:.4f}")


class ProjetAustralieModelisation:

    def __init__(self, data:pd.DataFrame):
        self.X=None
        self.y=None
        #self.data = data.dropna()
        self.data = data
        
        # s'il n'y a que mount ginini en climat 5, on degage
        if (self.data[self.data.Climat==5].Location.nunique()==1):
            self.data = self.data[self.data.Climat!=5]
            
        # palette
        palette_set1 = px.colors.qualitative.Set1
        self.palette=[]
        for i in range(7):
            self.palette.append(pc.unconvert_from_RGB_255(pc.unlabel_rgb(palette_set1[i])))
            

    def _modelisation_preparation(self, cible:str, scale:bool, climat:int=None, location:str=""):
        
        # filtrage eventuel
        data=self.data
        if climat!=None:
            data = self.data[self.data.Climat==climat]
        if location!="":
            data = self.data[self.data.Location==location]
        
            
        #self.X = data.drop(columns=cible)      
        self.Xy = data.copy()
        self.y = data[cible]
        
        # supprime toutes les infos sur la meteo future
        self.Xy = self.Xy.loc[:,~self.Xy.columns.str.startswith("Rain_J_")]
        self.Xy = self.Xy.loc[:,~self.Xy.columns.str.startswith("MaxTemp_J_")]
        self.Xy = self.Xy.loc[:,~self.Xy.columns.str.startswith("Rainfall_J_")]
        
        # on supprime la Location si elle est presente en str (utile uniquement pour filtrer en amont)
        if hasattr(self.Xy, "Location"):
            self.Xy = self.Xy.drop(columns="Location")
        
        # supprime les autres colonnes donnant trop d'indices
        if cible=="RainToday":
            self.Xy = self.Xy.drop(columns=["Rainfall"]) # si on veut predire RainToday, on supprime Rainfall, sinon c'est de la triche...
        if cible.startswith("Rain_J_"):
            self.Xy = self.Xy.drop(columns=["RainTomorrow"]) # si on veut predire RainToday, on supprime Rainfall, sinon c'est de la triche...

        #self.X = self.X.drop(columns=self.X.columns[self.X.columns.str.startswith('WindGustDir')])
        if cible.startswith("Wind"):
            self.Xy = self.Xy.drop(columns=self.Xy.columns[self.Xy.columns.str.startswith('Wind')])

        # on reinjecte la cible, on fait un dropna et on eclate entre X et y        
        self.Xy[cible] = self.y
        self.Xy = self.Xy.dropna()
        self.y = self.Xy[cible]
        self.X = self.Xy.drop(columns=cible)      
        
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
        #X_train, y_train = oversample.fit_resample(X_train, y_train)          
            
            
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        
    def _modelisation_matrice_confusion(self, clf):
        y_pred = clf.predict(self.X_test)
        print(pd.crosstab(self.y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))
        
        
    def modelisation(self, nom_modele:str="XGBoost", cible:str="RainTomorrow", gs:bool=True, climat:int=None, location:str="", totalite:bool=True):
               
        # seules les variables débutant par Rain impliquent de la classification, sauf Rainfall
        est_classification = cible.startswith("Rain") and not cible.startswith("Rainfall")
                
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
            if est_classification:
                existe_proba=True
                clf=xgb.XGBClassifier(random_state=0, learning_rate=.1, n_estimators=100, max_depth=4) 
                param_modele={ 'learning_rate': [0.1, 0.01, 0.001],
                              'n_estimators': [5, 10, 15, 20, 25],#, 50],
                              'max_depth': [3, 4, 5, 6],#, 7, 8, 9, 10],
                              }
            else:
                clf=xgb.XGBRegressor(random_state=0, learning_rate=.1, n_estimators=100, max_depth=4)

        elif nom_modele=='MLPClassifier':
            
            # max 1000, hidden 1200, alpha 0.0001 : 70 71 79 90 85 (170 mn)
            existe_proba=True
            # sgd = MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(1200,), alpha=0.0001) # 74%, 8mn / 97%             
            clf = MLPClassifier(random_state=5, max_iter=300, hidden_layer_sizes=(100,100), verbose=0) # 74%, 8mn / 97%
            param_modele = {'hidden_layer_sizes': [(50,200,50), (100,100), (100,)],
                            #'activation': ['tanh', 'relu'],
                            #'solver': ['sgd', 'adam']
                            
                            }
        
        elif nom_modele=='DummyClassifier':
            existe_proba=True
            clf=DummyClassifier(random_state=0, strategy='stratified')
            
        else:
            print("\n -------\nNom de modèle inconnu\n -------\n")
            return
        
        # gridsv
        if param_modele!=None and gs:
            outer_cv = StratifiedKFold(n_splits=3, shuffle=True)
            resc = make_scorer(recall_score,pos_label=1) # la difficulte est de predire correctemetnt les jours où il pleut reellement => il faut optimiser le recall sur la cible 1
            
            gcv = GridSearchCV(estimator=clf, 
                        param_grid=param_modele,
                        #scoring='recall',
                        #scoring = resc,
                        #scoring=score_callback,
                        scoring='roc_auc',
                        verbose=0,
                        cv=outer_cv,
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
        self.titre_modele = self.titre_graphe(nom_modele, hp, climat, location, cible)

        # n'execute la suite que si on veut le traitement total
        # pour des traitements successifs, on s'arrete là
        if not totalite:
            return

        
        print('Modele ', type(clf))
        
        predictions=clf.predict(self.X_train)
        i_fin_train=time.time()

        predictions=clf.predict(self.X_test)
        i_fin_test=time.time()

        # s'il s'agit de classfication:        
        if est_classification:
        
            print ("Precision train: {:.2f}% - Temps train: {:.2f} minutes".format(clf.score(self.X_train, self.y_train)*100, (i_fin_train-i_temps_debut)/60))
            print ("Precision test: {:.2f}% - Temps test: {:.2f} minutes".format(clf.score(self.X_test, self.y_test)*100, (i_fin_test-i_fin_train)/60))
    #        print (f"\nScore F1: {f1_score(self.y_test, predictions):.2f} - Accuracy: {accuracy_score(self.y_test, predictions):.2f} - Recall: {recall_score(self.y_test, predictions):.2f} - Precision: {precision_score(self.y_test, predictions):.2f}\n")
            
            
            self._modelisation_matrice_confusion(self.clf)
        
            predictions_proba=np.zeros(shape=predictions.shape)
            if existe_proba:
                predictions_proba=clf.predict_proba(self.X_test).max(axis=1)
              
            # renvoie les predictions sur le jeu complet
            #predictions=clf.predict(t_donnees_completes)
            #predictions_proba=np.zeros(shape=predictions.shape)
    #        if existe_proba:
    #╩            predictions_proba=clf.predict_proba(t_donnees_completes).max(axis=1)
            
            self.trace_courbe_roc(clf, self.titre_modele)

        # s'il s'agit de regression
        else:
            mse = mean_squared_error(self.y_test, predictions)
            mae = mean_absolute_error(self.y_test, predictions)
            print (f"\n ----- \n MSE : {mse:.2f} - MAE : {mae:.2f}\n ----- \n\n")

        i_fin_completes=time.time()
        print (" Temps comp: {:.2f} minutes".format( (i_fin_completes-i_fin_test)/60))
                
        print (time.ctime())
        
    def trace_courbe_roc(self, clf, titre_graphe:str=""):
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
        
            
        plt.title(f'Courbe ROC \n{titre_graphe}')
        plt.legend(loc='lower right')
        plt.show()
        
        y_pred_seuil = clf.predict_proba(self.X_test)[:,1] >= best_seuil
        print (f"\nScore F1: {f1_score(self.y_test, y_pred_seuil):.2f} - Accuracy: {accuracy_score(self.y_test, y_pred_seuil):.2f} - Recall: {recall_score(self.y_test, y_pred_seuil):.2f} - Precision: {precision_score(self.y_test, y_pred_seuil):.2f}\n")
        print (f"Matrice de confusion avec seuil de {best_seuil:.2f}:")
        print(pd.crosstab(self.y_test, y_pred_seuil, rownames=['Classe réelle'], colnames=['Classe prédite']))

        print(classification_report_imbalanced(self.y_test, y_pred_seuil))

    def titre_graphe(self, nom_modele:str, hp:str, climat:int=None, location:str="", cible:str=""):
        titre_clim_loc=""        
        if climat!=None:
            titre_clim_loc="\nClimat: "+str(climat)
        if location!="":
            titre_clim_loc="\nLocation: "+location
            
        return f'Modèle {nom_modele} \n {hp}{titre_clim_loc} \n Variable cible:{cible}'

    def AUC_nb_J(self, nom_modele:str="XGBoost", cible:str="RainTomorrow", gs:bool=False, climat:int=None, location:str="", nbj:int=8):
        scores_auc=[]
        
        for j in range(1,nbj):
            v_cible = f"Rain_J_{j:02d}"
            self.modelisation(nom_modele, v_cible, gs, climat, location, totalite=False )

            # calcule AUC
            y_test_pred_proba = self.clf.predict_proba(self.X_test)
            fpr, tpr, thresholds = roc_curve(self.y_test, y_test_pred_proba[:,1])
            roc_auc = auc(fpr, tpr)
            scores_auc.append(roc_auc)
            
        #self.scores_auc=scores_auc
        return scores_auc        


    def AUC_par_climat(self, nom_modele:str="XGBoost", cible:str="Rain", gs:bool=False, nbj:int=8):
        all_scores_auc=[]
        climats=[]
        for climat in self.data.Climat.unique():
            all_scores_auc.append(self.AUC_nb_J(nom_modele, cible, gs=gs, climat=climat, nbj=nbj))
            climats.append(climat)
            
        self.AUC_trace(all_scores_auc, climats, nbj=nbj)

    def AUC_par_location(self, nom_modele:str="XGBoost", cible:str="Rain", gs:bool=False, climat:int="", nbj:int=8):
        all_scores_auc=[]
        locations=[]       
        
        # si un climat est defini, on affichera les villes de ce climat uniquement
        data = self.data
        if climat!="":
            liste_locations=self.data[self.data.Climat==climat].Location.unique()
        else:
            liste_locations=self.data.Location.unique()
        
        for location in liste_locations:
            all_scores_auc.append(self.AUC_nb_J(nom_modele, cible, gs=gs, location=location, nbj=nbj))
            locations.append(location)
            
        self.AUC_trace(all_scores_auc, locations, mode="Location", nbj=nbj)

        
    def AUC_trace(self, scores_auc, types, mode:str="Climat", nbj:int=8):
        fig = plt.figure(figsize=(12,8))
        
        if mode=="Climat":
            for score_auc, item_type in zip(scores_auc, types):
                plt.plot(range(1,nbj), score_auc, label=f"Climat {item_type}", color=self.palette[item_type])
        else:
            for score_auc, item_type in zip(scores_auc, types):
                plt.plot(range(1,nbj), score_auc, label=f"{item_type}")
            
        plt.ylabel("AUC")
        plt.xlabel("Numéro de la journée à J+n prédite pour RainToday")
        plt.title("Score AUC en fonction du décalage de prédiction de pluie dans le futur\n")
        plt.legend(loc='upper right')
        plt.ylim(.45,1)
        plt.axhline(y=0.5, color='gray', linestyle='dashed')
        plt.show();
        
    # regressions
    
    def MSE_nb_J(self, nom_modele:str="XGBoost", cible:str="MaxTemp", gs:bool=False, climat:int=None, location:str="", nbj:int=8):
        scores_mse=[]
        scores_mae=[]
        
        for j in range(1,nbj):
            v_cible = f"{cible}_J_{j:02d}"
            self.modelisation(nom_modele, v_cible, gs, climat, location, totalite=False )

            predictions=self.clf.predict(self.X_test)
            mse = mean_squared_error(self.y_test, predictions)
            mae = mean_absolute_error(self.y_test, predictions)

            scores_mse.append(mse)
            scores_mae.append(mae)
            
        return scores_mse, scores_mae        

    
    def MSE_par_climat(self, nom_modele:str="XGBoost", cible:str="MaxTemp", gs:bool=False, nbj:int=8):
        all_scores_mse=[]
        all_scores_mae=[]
        climats=[]
        for climat in self.data.Climat.unique():
            mse, mae = self.MSE_nb_J(nom_modele, cible, gs=gs, climat=climat, nbj=nbj)
            all_scores_mse.append(mse)
            all_scores_mae.append(mae)
            
            climats.append(climat)
            
        self.scores_trace(all_scores_mse, climats, nbj=nbj, cible=cible, libelle="MSE")
        self.scores_trace(all_scores_mae, climats, nbj=nbj, cible=cible, libelle="MAE")
        
    def MSE_par_location(self, nom_modele:str="XGBoost", cible:str="MaxTemp", gs:bool=False, climat:int="", nbj:int=8):
        all_scores_mse=[]
        all_scores_mae=[]
        locations=[]       
        
        # si un climat est defini, on affichera les villes de ce climat uniquement
        data = self.data
        if climat!="":
            liste_locations=self.data[self.data.Climat==climat].Location.unique()
        else:
            liste_locations=self.data.Location.unique()
        
        for location in liste_locations:
            mse, mae = self.MSE_nb_J(nom_modele, cible, gs=gs, location=location, nbj=nbj)
            all_scores_mse.append(mse)
            all_scores_mae.append(mae)
            
            locations.append(location)
            
        self.scores_trace(all_scores_mse, locations, mode="Location", nbj=nbj, cible=cible, libelle="MSE")
        self.scores_trace(all_scores_mae, locations, mode="Location", nbj=nbj, cible=cible, libelle="MAE")

        
        
    def scores_trace(self, scores, types, mode:str="Climat", nbj:int=8, cible:str="MaxTemp", libelle="MSE"):
        fig = plt.figure(figsize=(12,8))

        if mode=="Climat":
            for score, item_type in zip(scores, types):
                plt.plot(range(1,nbj), score, label=f"Climat {item_type}", color=self.palette[item_type])
        else:
            for score, item_type in zip(scores, types):
                plt.plot(range(1,nbj), score, label=f"{item_type}")
            
        plt.ylabel(libelle)
        plt.xlabel(f"Numéro de la journée à J+n prédite pour {cible}")
        plt.title(f"{libelle} en fonction du décalage de prédiction dans le futur\n")
        plt.legend(loc='upper right')
        #plt.ylim(.45,1)
        #plt.axhline(y=0.5, color='gray', linestyle='dashed')
        plt.show();              
        
    # ---- series temporelles
    
    def prepare_serie_temporelle(self, location:str="", variable:str="MaxTemp", affiche=True):
        df = self.data
        if location!="":
            df = self.data.loc[self.data.Location==location]

        # on ne reprend pas plus tôt à cause des trous sur 3 mois
        df = df.loc[df.index>='2013-03-03']
        #df = df.loc[df.index>='2009-01-01']

        self.serie_temporelle=df[variable]
        self.titre_analyse = location+str(" - ")+variable
        
        if affiche:
            plt.figure(figsize=(16,8))
            plt.plot(self.serie_temporelle)
            plt.title(self.titre_analyse)
        
    def decompose_serie_temporelle(self):
        result = seasonal_decompose(self.serie_temporelle, model='additive', period=365)
        result.plot();

    def affiche_acf_pacf(self):                                 # darwin - mildura
        plot_acf(self.serie_temporelle.diff(1).dropna(), lags = 30)     # 3 - 4
        plot_acf(self.serie_temporelle.diff(365).dropna(), lags = 30)   # 10- 3

        plot_pacf(self.serie_temporelle.diff(1).dropna(), lags = 30)    # 5 - 12
        plot_pacf(self.serie_temporelle.diff(365).dropna(), lags = 30)  # 2 - 2
                       
    
    def applique_sarima(self, p=2, d=1, q=0, P=0, D=1, Q=1, ax=None):
        
        # tronconne la série
        serie_temporelle_debut = self.serie_temporelle.iloc[:-365]
        #serie_temporelle_fin = self.serie_temporelle.iloc[-12:]
        
        # SARIMAX
        print (time.ctime())

        smx = sm.tsa.SARIMAX(serie_temporelle_debut, order=(p,d,q), seasonal_order=(P,D,Q,365))
        smx_fitted = smx.fit()#maxiter=1000)

        print (time.ctime())

        print(smx_fitted.summary())
        self.smx = smx_fitted
        
#        pred = smx_fitted.predict(49, 61)
#        serie_predite = pd.concat([serie_temporelle_debut, pred])

#        self.sp = serie_predite

#        plt.figure(figsize=(16,8))
#        plt.plot(serie_predite)
#        plt.axvline(x=datetime.date(2020,12,15), color='orange')

        # avec intervalle de confiance
        
        prediction = smx_fitted.get_forecast(steps =365).summary_frame()  #Prédiction avec intervalle de confiance
        
        if ax==None:
            fig, ax = plt.subplots(figsize = (40,5))
            
        ax.plot(self.serie_temporelle)
        prediction['mean'].plot(ax = ax, style = 'k--') #Visualisation de la moyenne
        ax.fill_between(prediction.index, prediction['mean_ci_lower'], prediction['mean_ci_upper'], color='k', alpha=0.1); #Visualisation de l'intervalle de confiance    
        
        # affiche N-1
        last_12_months = self.serie_temporelle.shift(365)
        ax.plot(last_12_months[-365:], label="N-1")      
        
        ax.set_title(self.titre_analyse)
                
    # -----    
    # a deplacer en dataviz
    def animation_variable(self, variable:str="RainToday", discrete:bool=False):
        
        data = self.data.loc[(self.data.index>='2014-04-01')&(self.data.index<='2014-04-30'),:].copy()
        data["Date"] = data.index
        
        if discrete:
            cible = data[variable].astype(str)
        else:
            cible = data[variable]
        
        fig = px.scatter_mapbox(data, 
                                lat='lat', 
                                lon='lng', 
                                hover_name='Location', 
                                color=cible, 
                                #text='Location', 
                                #labels=clf.labels_, 
                                animation_frame="Date",
#                                animation_group="Location",
                                size_max=30, 
                                opacity=.8,
                                #color_continuous_scale=px.colors.qualitative.Plotly
                                color_discrete_sequence=px.colors.qualitative.Set1,
                                #color_discrete_sequence=px.colors.qualitative.T10,
                                range_color=[data[variable].min(), data[variable].max()]
                                ).update_traces(marker=dict(size=30))
                
        fig.update_layout(mapbox_style='open-street-map')
        fig.show(renderer='browser')      

#pm = ProjetAustralieModelisation(pd.read_csv("data_basique.csv", index_col=0))
pm = ProjetAustralieModelisation(pd.read_csv("data_process4_knnim_resample_J365.csv", index_col=0))

#pm.animation_variable()

# data_process_non_knnim => preprocession avancée (mais sans knni ni reequilibrage des classes)
# data_process4_knnim_resample_J365 => idem, mais 365j de prevision pour Rainfall, MaxTemp, RainToday. Knni SANS drop RainTomorrow (car plein de variables cibles possibles), SaisonCos
# data_process3_knnim_resample_J2 => version light
