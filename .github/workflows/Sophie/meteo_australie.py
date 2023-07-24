# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 21:13:22 2023

@author: Sophie
"""
# ajout d'un commentaire pour git
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import chi2_contingency

class ProjetAustralie:
    
    def __init__(self):
        self.charge_donnees()
        self.preprocessing()
        
    def charge_donnees(self):
        self.df = pd.read_csv("weatherAUS.csv")
        self.df.info()
        print(self.df.head())   
    
    def preprocessing(self):
        # remplace Yes/No par 0/1
        self.df = self.df.replace({"No":0, "Yes":1})
        self.df.Date = pd.to_datetime(self.df.Date)
    
    def preprocessing_apres_analyse(self):
        # ajout de AmplitudeTemp
        self.df["AmplitudeTemp"]=self.df.MaxTemp - self.df.MinTemp
        
        # suppression des variable très fortement correlées
        self.df = self.df.drop(columns=["MinTemp", "Temp9am", "Temp3pm", "Pressure3pm"])
        
        
    def analyse_donnees(self):
        # villes
        dlc = self.df.Location.value_counts()
        print (f"il y a {len(dlc)} villes")
        print (dlc)
        
        # 3 villes (Uluru, Katherine, Nhil) ne contiennent environ que moitié moins de relevés (1578) que les autres villes, qu ien ont entre 3001 et 3436
        #> => drop? regarder leur localisation géographique pour voir si une ville à proximité existe et s'il y a une forte corrélation sur la météo pourles jours en commun
               
        
        # colonnes avec beaucoup de NA
        print ("Colonnes avec NA")
        print (self.df.isna().sum()/self.df.shape[0])
        # Evaporation, Sunshine, Cloud9am, Cloud3pm contiennent beaucoup de NA (39% pour Cloud9am jusqu'à 48% pour Sunshine)
        # => drop? regarder corrélation de cette variable avec les temperatures et la pluie lorsqu'elle est disponible

        plt.figure(figsize=(12,6))
        plt.bar(self.df.columns, self.df.isna().sum() )
        plt.xticks(rotation=90);
        
        # verification des correlations entre les variables numeriques
        # => correlation intéressante entre RainTomorrow et RainToday, Cloud3pm (dommage...), Cloud9am (idem...), 
        #   Humidity3pm, Humidity9am, WindGustSpeed, Rainfall
        df_num = self.df.select_dtypes(include=['float64'])
        
        # affichage des correlations 
        fig, axes = plt.subplots(1,2,figsize=(24,12))
        
        # correlations fortes
        # => aucune vairable n'est fortement correlee avec RainTomorrow
        hm = sns.heatmap(abs(df_num.corr())>.5, cmap="binary", ax=axes[0])
        axes[0].set_title("Corrélations fortes (>.5)", fontsize=18)
        
        # toutes correlations
        # => correlations intéressantes (.25<corr<.5) avec Sunshine, Humidity3pm, Humidity9am, Cloud9am (dommage...), Cloup9pm (idem), RainToday
        cmap_25 = sns.color_palette(['red', 'orange', 'yellow', 'white', 'white', 'yellow', 'orange', 'red'])
        sns.heatmap(df_num.corr(), cmap=cmap_25, annot=True, fmt=".1f", ax=axes[1], vmin=-1)
        axes[1].set_title("Corrélations", fontsize=18)
        
        print(df_num.corr())
        
        plt.show();
        # MaxTemps est très fortement corrélé avec MinTemps (.7), Temp9am (.9), Temp3pm (1)=> envisager d'en supprimer
        # Idem pour Pressure3pm très fortement correlé avec Pressure9am (1)
        
        # -------------------------------------------

        plt.figure(figsize=(24,24))
        sns.relplot(data=self.df, y="MaxTemp", x="MinTemp", hue="RainTomorrow")
        plt.show();
        # La relation entre MinTemps et MaxTemps indique que la variable RainTomorrow se trouve très fréquement sur la 1ère bissectrice
        # => avant de supprimer MinTemps, qui est redondante avec les autres variables de température vu les corrélations calculées,
        # nous allons ajouter une nouvelle variable AmplitudeTemp, égale à MaxTemp-MinTemp, afin d'aider le modèle

        # -------------------------------------------

        fig, axes = plt.subplots(1,1,figsize=(24,12))
        
        # affiche le nb de NA pour chaque variable et pour chaque Location
        df_nb_na = self.df.groupby('Location').apply(lambda x: x.isna().sum()).drop(columns=["Location"])
        df_nb = self.df.groupby('Location').size().reset_index(name='NbTotEnregistrements\n(nuls ou non)')
        
        df_nb_na = df_nb_na.merge(df_nb, left_index=True, right_on='Location').set_index("Location")
        
        self.df_nb_na=df_nb_na
        
        print("\n Nb de valeurs nulles par Location pour chaque variable\n")
        print (df_nb_na.iloc[:,1:])
        
        sns.heatmap(df_nb_na.iloc[:,1:], cmap='gnuplot', annot=True, fmt="d")
        axes.set_title("Nb d'enregistrements nuls par Location", fontsize=18)
        plt.show();
        
        # trous dans le temps
        plt.figure(figsize=(48,12))
        #plt.xticks(pd.date_range(start='2007-01-01', end='2018-12-31', freq='M'), rotation=90)
        self.df.groupby([pd.Grouper(key='Date', freq='w'), 'Location']).size().unstack().plot(kind='line', figsize=(24, 16))
        #df_tmp = self.df.groupby([pd.Grouper(key='Date', freq='M'), 'Location']).size().unstack()
        #plt.plot(df_tmp)
        #plt.legend()
        plt.show();
    
    
    # in progress #
    def indexation_temporelle_complete(self):
        date_min = self.df.Date.min()
        date_max = self.df.Date.max()
        print ("Plage de date: du ", date_min, "au ", date_max)
        print (date_max-date_min, "jours")
        
        date_range = pd.date_range(start=self.df.Date.min(), end=self.df.Date.max(), freq='D')     
        
        
    # analyse temporelle - à approfondir
    def analyses_temporelles(self):
        self.df["DayOfYear"] = self.df.Date.dt.dayofyear
        self.df["Week"] = self.df.Date.dt.isocalendar().week
        self.df["Year"] = self.df.Date.dt.year
        
        df_tmp = self.df.loc[self.df.Location=="Melbourne"].groupby(["Week", "Year"]).agg({"MaxTemp":"mean"}).unstack()
    
        plt.close('all')
        fig, axes = plt.subplots(1,2,figsize=(16,8))
        #sns.lineplot(data=df_tmp, x='DayOfYear', y='Rainfall', hue='Year')
        plt.subplot(121)

        plt.title("Température moyenne hebdomadaire (MaxTemp) - Melbourne")
        plt.xlabel("Numéro de semaine")
        plt.ylabel("MaxTemp (°)")
        
        for col in df_tmp.columns:
            plt.plot(df_tmp.index, df_tmp[col], label=col[1])
        plt.legend()
        
        sns.lineplot(data=pa.df[pa.df.Location=="Melbourne"], x="Date", y="MaxTemp", hue="Year", ax=axes[1])
        axes[1].set_title("Evolution de MaxTemp pour la ville de Melbourne")
        axes[1].set_xlabel("Année")
        axes[1].set_ylabel("MaxTemp (°)")

        plt.show();
    
    # -----------------------------
    # calcule la correlation avec chaque variable qualitative du vent
    def correlation_vent(self):
        self.df_chi2=pd.DataFrame()
        var_vent=["WindGustDir", "WindDir9am", "WindDir3pm"]
        for e in var_vent:
            self._correlation_var(e)
        print("\np-value issu du Chi2 entre les variables du vent et RainTomorrow:\n",self.df_chi2)
    
        print ("\n Villes ayant au moins une p-value >.05\n", self.df_chi2[self.df_chi2.max(axis=1)>.05])
    
    # calcule le chi2 pour une variable qualitative
    def _correlation_var(self, var):
        self.df_chi2[var]=""

        # calcule la correlation du vent, toutes villes confondues        
        dfl=self.df
        self.df_chi2.loc["*Tout",var] = self._correlation_vars(dfl, "RainTomorrow", var) 
        
        # calcule les correlations du vent, par ville
        for e in self.df.Location.unique():
            dfl = self.df.loc[self.df.Location==e]
            #print (e, var)
            self.df_chi2.loc[e,var]=self._correlation_vars(dfl, "RainTomorrow", var) 
            
        self.df_chi2 = self.df_chi2.sort_values(var, ascending=False)
        #self.chi2=df_chi2
        
    # calcule le chi2 entre deux variables
    # renvoie -1 si aucune occurence n'existe pour le couple
    def _correlation_vars(self, df, var1, var2):
            tcd = pd.crosstab(df[var1], df[var2])
            if (len(tcd)>0):
                res_chi2 =chi2_contingency(tcd)
                return res_chi2[1]
            return -1
        
    # -----------------------------
    
    # in progress: j'aimerais tenter d'arriver à clusteriser automatiquement les villes
    def clusterisation(self):
        # tentative de determination de climats similaires dans certaines villes
        from sklearn.cluster import KMeans
        
        df_cluster = self.df.dropna(how='any')
        df_cluster_f = df_cluster.select_dtypes(include=['float64'])
        
        df_cluster['Location_ID'] = df_cluster.groupby('Location').ngroup()
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(df_cluster_f, sample_weight=df_cluster['Location_ID'])
        df_cluster['ClusterLocation'] = kmeans.labels_
        print(df_cluster[['Location', 'ClusterLocation']])
    
    # -----------------------------
    # représentation geographique sur une carte    
    def synthetise_villes(self):
        #s_rt = self.df.groupby("Location")["RainTomorrow"].mean()
        df_rt = self.df.groupby("Location").agg({"RainTomorrow":'mean', "MaxTemp":'mean', "Pressure9am":'mean', 'Location':'count'})
        df_rt = df_rt.rename(columns={"Location":"Nb"})
        
        df_rt = df_rt.merge(self.df_cities, left_index=True, right_on='Location')
        
        fig = px.scatter_mapbox(df_rt, lat='lat', lon='lng', hover_name='Location', color='MaxTemp', size='RainTomorrow')
        fig.update_layout(mapbox_style='open-street-map')
        #fig.update_layout(margin={'r':0, 't':0, 'l':0, 'b':0})
        
        fig.show(renderer='browser')
              
    
    # chargement des données geographiques
    def carte_australie(self):
        
        # Créer un DataFrame avec les coordonnées de la ville de Paris
        self.df_cities = pd.read_csv("villes_coordonnees.csv",sep=";")     
                
        # Afficher la carte avec Plotly Express
        fig = px.scatter_mapbox(self.df_cities, lat='lat', lon='lng', hover_name='Location', color='lat', size=self.df_cities.lng/10-10, text='Location', labels='Location')
        fig.update_layout(mapbox_style='open-street-map')
        fig.update_layout(margin={'r':0, 't':0, 'l':0, 'b':0})
        fig.show(renderer='browser')


pa = ProjetAustralie()

pa.analyse_donnees()
#pa.preprocessing_apres_analyse()
pa.carte_australie()
pa.synthetise_villes()