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
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

class ProjetAustralie:
    
    def __init__(self):
        self.charge_donnees()
        self.preprocessing()
        self.is_preprocessing_apres_analyse=False
        
    def charge_donnees(self):
        self.df = pd.read_csv("weatherAUS.csv")
        self.df.info()
        print(self.df.head())   
    
    def preprocessing(self):
        # remplace Yes/No par 0/1
        self.df = self.df.replace({"No":0, "Yes":1})
        self.df.Date = pd.to_datetime(self.df.Date)
        self.df = self.df.set_index("Date", drop=False)
    
    def preprocessing_apres_analyse(self):
        # si deja fait, on sort
        if self.is_preprocessing_apres_analyse:
            return
        
        # ajout de AmplitudeTemp
        self.df["AmplitudeTemp"]=self.df.MaxTemp - self.df.MinTemp
        
        # suppression des variable très fortement correlées
        self.df = self.df.drop(columns=["MinTemp", "Temp9am", "Temp3pm", "Pressure3pm"])
        
        # supprime la date (dispo en index)
        self.df = self.df.drop(columns="Date")

        # ajoute les proprietes des villes
        self._ajoute_prop_locations()
        
        # supprime le nom de la ville, puisqu'on a ses coordonnées => pas de dummies sur le nom de la ville
        self.df = self.df.drop(columns="Location")       
        
        # get_dummies (variables sur le vent)
        self.df= pd.get_dummies(self.df)
        
        # retrait "bourrin" des NA (à affiner plus tard)
        self.df = self.df.dropna()
        
        # indique qu'on a deja fait cette etape
        self.is_preprocessing_apres_analyse=True
    
    # ajoute les proprietes des villes au DF
    def _ajoute_prop_locations(self):
        # si la colonne de la latitude a déja été ajoutée, on sort
        if 'lat' in self.df.columns:
            return
        # sinon, on charge les props des villes et on les ajoute au df
        self.charge_villes()
        self.df = pd.merge(self.df, self.df_cities, on='Location')
                
    # remplace RainTomorrow par 'Rain+j', qui indique s'il pleuvra j jours plus tard
    # !! non fonctionnelle encore !!
    def _remplace_rain_tomorrow(self, j):
        df["DateFuture"] = df.index + pd.DateOffset(days=1)
        for e in self.df.Location.unique():
            self.df.loc[:,"Location"==e].RainTomorrow = self.df.apply(lambda row: self.df.loc[self.df.index == row['DateFuture'], "Location"==e].RainToday, axis=1)
        
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
        
    # --------------------------------------------
    # affiche les graphes representant les NA et les moyennes en fonction du temps, pour chaque colonne numerique
    def analyse_variables_temps(self):
        for e in self.df.select_dtypes(include=['float64']).columns:
            pa.analyse_variable_temps(e, "M")        
            
    # --------------------------------------------
    # affiche les graphes representant les NA et les moyennes en fonction du temps
    def analyse_variable_temps(self, variable, frequence):
        
        # si l'attribut n'a pas encore été créé, alors on fait la reindexation temporelle
        if ~hasattr(self, "df_resample"):
            self.reindexation_temporelle()
            
        df = self.df_resample[['Location', variable]]       
        df["NbNA"] = df[variable].isna()
        gdf = df.groupby([pd.Grouper(freq=frequence), "Location"]).agg({variable:'mean', 'NbNA':'sum'}).reset_index()
        gdf = gdf.rename(columns = {"level_0":"Temps"})        
        
        self.gdf = gdf
        
        # NA
        fig, axes = plt.subplots(2,2,figsize=(36,18))

        sns.lineplot(data=gdf, x="Temps", y="NbNA", ax=axes[0,0])
        axes[0,0].set_title("Nombre de NA de "+variable+" (toutes localités confondues)", fontsize=18)
        axes[0,0].set_xlabel("Temps (frequence: "+frequence+")")
        sns.lineplot(data=gdf, x="Temps", y="NbNA", hue='Location', ax=axes[0,1], legend=None)
        axes[0,1].set_title("Nombre de NA "+variable+" (par localité)", fontsize=18)
        axes[0,1].set_xlabel("Temps (frequence: "+frequence+")")
       
        # Moyenne des valeurs
        sns.lineplot(data=gdf, x="Temps", y=variable, ax=axes[1,0])
        axes[1,0].set_title("Moyenne de "+variable+" (toutes localités confondues)", fontsize=18)
        axes[1,0].set_xlabel("Temps (frequence: "+frequence+")")
        sns.lineplot(data=gdf, x="Temps", y=variable, hue='Location', ax=axes[1,1], legend=None)
        axes[1,1].set_title("Moyenne de "+variable+" (par localité)", fontsize=18)
        axes[1,1].set_xlabel("Temps (frequence: "+frequence+")")

        fig.show()
        
        # Tracer la courbe avec Plotly
        fig = px.line(gdf, x="Temps", y="NbNA", color='Location')
        
        # Ajouter les titres et les labels
        fig.update_layout(
            title="Nombre de NA de "+variable+" (toutes localités confondues)",
            xaxis_title="Temps (frequence: "+frequence+")",
            yaxis_title="NbNA"
        )        
        # Afficher le graphique
        fig.show(renderer='browser')

        
    
    # reindexe les dates pour qu'il n'y ait aucun trou
    def reindexation_temporelle(self):
        date_min = self.df.index.min()
        date_max = self.df.index.max()
        print ("Plage de date: du ", date_min, "au ", date_max)
        print (date_max-date_min)
        
        # cree un range de la date min à la max
        # Un 'resample' ne suffit pas car les dates min et max doivent etre communes à chaque Location
        date_range = pd.date_range(start=date_min, end=date_max, freq='D')     
        date_range =pd.DataFrame(index=date_range)
        
        df_dates = pd.DataFrame()
        
        for location in self.df.Location.unique():
            df_l = self._reindexation_temporelle_location(location, date_range)
            df_dates = pd.concat([df_dates, df_l], axis=0)
            
        df_dates['NbTotNA'] = df_dates.iloc[:,1:].isna().sum(axis=1)
        self.df_resample = df_dates
        
    # renvoie un df de la location reechantilloné sur une plage de date passée en argument
    def _reindexation_temporelle_location(self, location, date_range):
        df_l = self.df[self.df.Location==location]
        df_l_c = pd.concat([date_range, df_l], axis=1)
        df_l_c.Location = location # nécessaire car les nouvelles dates entrainent une location nulle
        return df_l_c
        
    # comparaison des NA à partir du DF d'origine et celui resamplé
    def comparaison_avec_sans_dates_reindexees(self, location, variable, frequence):
        # si l'attribut n'a pas encore été créé, alors on fait la reindexation temporelle
        if ~hasattr(self, "df_resample"):
            self.reindexation_temporelle()

        fig, axes = plt.subplots(2,1,figsize=(24,12))
        # sans reindexation
        self._comparaison_avec_sans_dates_reindexees(self.df[['Location', variable]], location, variable, frequence, axes[0])
        axes[0].set_title("Nombre de NA de "+variable+" à "+location+" \n(sans resample sur les dates)", fontsize=18)
        axes[0].set_xlabel("")

        # avec reindexation
        self._comparaison_avec_sans_dates_reindexees(self.df_resample[['Location', variable]], location, variable, frequence, axes[1])
        axes[1].set_title("Nombre de NA de "+variable+" à "+location+" \n(avec resample sur les dates)", fontsize=18)
        axes[1].set_xlabel("Temps (frequence: "+frequence+")")
        fig.show()
        
    def _comparaison_avec_sans_dates_reindexees(self, df, location, variable, frequence, axe):

        df=df[df.Location==location]
        
        df["NbNA"] = df[variable].isna()
        gdf = df.groupby([pd.Grouper(freq=frequence), "Location"]).agg({variable:'mean', 'NbNA':'sum'}).reset_index()
        gdf = gdf.rename(columns = {"level_0":"Temps", "Date":"Temps"})        
        self.ggdf=gdf
        # NA
        sns.lineplot(data=gdf, x="Temps", y="NbNA", ax=axe)
        #axe.set_xlabel("Temps (frequence: "+frequence+")")

        
        
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
    # -----------------------------
    # -----------------------------
    
    # in progress: j'aimerais tenter d'arriver à clusteriser automatiquement les villes
    def clusterisation_temporelle(self):
        # tentative de determination de climats similaires dans certaines villes
        from sklearn.cluster import KMeans
        
        df_cluster = self.df.dropna(how='any')
        df_cluster_f = df_cluster.select_dtypes(include=['float64'])
        
        df_cluster['Location_ID'] = df_cluster.groupby('Location').ngroup()
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(df_cluster_f, sample_weight=df_cluster['Location_ID'])
        df_cluster['ClusterLocation'] = kmeans.labels_
        print(df_cluster[['Location', 'ClusterLocation']])

    # clusterisation des villes en 7 zones climatiques, basées sur la moyenne des variables sur les 10 ans de relevés
    def clusterisation_groupee(self):
        from sklearn.cluster import AgglomerativeClustering, MeanShift, estimate_bandwidth
        from scipy.cluster.hierarchy import linkage, dendrogram

        self._ajoute_prop_locations()

        self.df_moyenne = self.df.groupby("Location").agg("mean")
        self.df_moyenne = self.df_moyenne.dropna(axis=1)

        # sur la suite, la normalisation et la clusterization ne se font pas sur les deux dernières colonnes,
        # c'est à dire les coordonnées geographiques, pour que celles-ci n'influence pas l'appartenance à un cluster
        scaler=MinMaxScaler()
        pa.df_moyenne.iloc[:,:-2] = scaler.fit_transform(pa.df_moyenne.iloc[:,:-2])        

        Z = linkage(self.df_moyenne.iloc[:,:-2], method='ward', metric = 'euclidean')
        
        plt.figure(figsize=(12,8))
        dendrogram(Z, labels=self.df_moyenne.index, leaf_rotation=90., color_threshold=1.25)
        plt.show()

        # 7 clusters (pas optimal, juste pour voir)        
        clf = AgglomerativeClustering(n_clusters=7)
        clf.fit(self.df_moyenne.iloc[:,:-2])
        self.clust_lab = clf.labels_

        #self.charge_villes()
        self.df_moyenne = self.df_moyenne.reset_index()
        
        fig = px.scatter_mapbox(self.df_moyenne, lat='lat', lon='lng', hover_name='Location', color=clf.labels_, text='Location', labels='Location', size_max=15, size='RainTomorrow', color_continuous_scale=px.colors.qualitative.Plotly).update_traces(marker=dict(size=10))
        fig.update_layout(mapbox_style='open-street-map')
        fig.show(renderer='browser')      
   
    # -----------------------------
    # -----------------------------
    # -----------------------------
    # représentation geographique sur une carte    
    def synthetise_villes(self):
        self.charge_villes()
        
        #s_rt = self.df.groupby("Location")["RainTomorrow"].mean()
        df_rt = self.df.groupby("Location").agg({"RainTomorrow":'mean', "MaxTemp":'mean', "Pressure9am":'mean', 'Location':'count'})
        df_rt = df_rt.rename(columns={"Location":"Nb"})
        
        df_rt = df_rt.merge(self.df_cities, left_index=True, right_on='Location')
        
        fig = px.scatter_mapbox(df_rt, lat='lat', lon='lng', hover_name='Location', color='MaxTemp', size='RainTomorrow')
        fig.update_layout(mapbox_style='open-street-map')
        #fig.update_layout(margin={'r':0, 't':0, 'l':0, 'b':0})
        
        fig.show(renderer='browser')             

    # charge infos sur villes
    def charge_villes(self):
        # si deja chargé, on sort
        if hasattr(self, "df_cities"):
            return
        # Créer un DataFrame avec les coordonnées de la ville de Paris        
        self.df_cities = pd.read_csv("villes_coordonnees.csv",sep=";")     
        
    
    # chargement des données geographiques
    def carte_australie(self):
        self.charge_villes()
                
        # Afficher la carte avec Plotly Express
        fig = px.scatter_mapbox(self.df_cities, lat='lat', lon='lng', hover_name='Location', color='lat', size=self.df_cities.lng/10-10, text='Location', labels='Location')
        fig.update_layout(mapbox_style='open-street-map')
        fig.update_layout(margin={'r':0, 't':0, 'l':0, 'b':0})
        fig.show(renderer='browser')

    # -----------------------------
    # -----------------------------
    # --- Modelisation    ---
    # -----------------------------
    # -----------------------------

    def _modelisation_preparation(self, scale):
        
        # si pas deja lancé
        self.preprocessing_apres_analyse()
        
        X = self.df.drop(columns="RainTomorrow")
        y = self.df.RainTomorrow

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66) 
        
        if scale:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        

    def _modelisation_matrice_confusion(self, clf):
        y_pred = clf.predict(self.X_test)
        print(pd.crosstab(self.y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))
        
        
    def modelisation_knn(self):
        from sklearn.neighbors import KNeighborsClassifier 
   
        clf_knn = KNeighborsClassifier(n_neighbors=5)              
        self._modelisation_preparation(False)
        
        clf_knn.fit(self.X_train, self.y_train)
        self._modelisation_matrice_confusion(clf_knn)


    def modelisation_random_forest(self):
        from sklearn.ensemble import RandomForestClassifier

        clf_rf = RandomForestClassifier(n_jobs=-1, random_state=66 )
        self._modelisation_preparation(False)
        
        clf_rf.fit(self.X_train, self.y_train)
        self._modelisation_matrice_confusion(clf_rf)
        
    

pa = ProjetAustralie()

pa.analyse_donnees()
#pa.preprocessing_apres_analyse()
#pa.carte_australie()
#pa.synthetise_villes()