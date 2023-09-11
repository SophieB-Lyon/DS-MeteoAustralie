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
from sklearn.impute import KNNImputer


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

        # matrice avant
        self.matrice_corr_quyen(self.df.select_dtypes(include=['float64']))

        # determine les climats
        self.clusterisation_groupee()
        self.df = self.df.merge(self.df_climat, on="Location")      
        
        # ajout de AmplitudeTemp
        self.df["AmplitudeTemp"]=self.df.MaxTemp - self.df.MinTemp
        
        # suppression des variable très fortement correlées et de celles trop manquantes
        self.df = self.df.drop(columns=["MinTemp", "Temp9am", "Temp3pm", "Pressure3pm", "Sunshine", "Evaporation", "Cloud9am", "Cloud3pm"])              
        
        # supprime la date (dispo en index)
        self.df = self.df.drop(columns="Date")

        # ajoute les proprietes des villes
        self._ajoute_prop_locations()
        
        # gestion des variables sur le vent
        #self.df= pd.get_dummies(self.df)
        self.remplace_direction_vent()
        
        # supprime variables redondantes du vent
        self.df = self.df.drop(columns=["WindGustDir_RAD", "WindDir9am_RAD", "WindDir3pm_RAD", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"])       
        
        # retrait "bourrin" des NA (à affiner plus tard)
        self.df = self.df.dropna()
                
        # supprime le nom de la ville, puisqu'on a ses coordonnées => pas de dummies sur le nom de la ville
        self.df = self.df.drop(columns="Location")              
        
        # affiche matrice de correlation apres
        self.matrice_corr_quyen(self.df.drop(columns="Climat"))
        
        for clim in np.arange(7):
            self.matrice_corr_quyen(self.df[self.df.Climat==clim].drop(columns="Climat"), "Corrélations climat n° "+str(clim))
        
        # indique qu'on a deja fait cette etape
        self.is_preprocessing_apres_analyse=True
    
    # affiche matrice corr comme Quyen
    def matrice_corr_quyen(self, df, titre:str="Corrélations entre variables après ajout des nouvelles variables"):
        cmap = sns.diverging_palette(260, 20, as_cmap=True)

        fig_corr, ax = plt.subplots(figsize=(12,12))
        corr_mat = df.corr()
        mask = np.triu(np.ones_like(corr_mat))

        sns.heatmap(corr_mat,
                    mask=mask,
                    annot=True,
                    fmt='.2f',
                    cmap=cmap,
                    ax=ax)
        ax.set_title(titre, fontsize=20)
        
    # remplace les variables categorielles de direction de vent par les composantes x et y
    def remplace_direction_vent(self):
        self._remplace_direction_vent("WindGustDir", "WindGustSpeed")
        self._remplace_direction_vent("WindDir3pm", "WindSpeed3pm")
        self._remplace_direction_vent("WindDir9am", "WindSpeed9am")       
    
    # remplace une colonne categorielle de direction de vent par deux colonnes numeriques
    def _remplace_direction_vent(self, nom_colonne_dir: str, nom_colonne_speed: str):
        df_direction = pd.DataFrame()
        df_direction["dir"]=["E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE"]
        
        # increment pour chacune des directions (sens horaire)
        increment=-np.pi/8
        
        df_direction["rad"]=increment*df_direction.index
        df_direction["sin"]=np.sin(df_direction.rad)
        df_direction["cos"]=np.cos(df_direction.rad)
        
        # jointure pour deduire les cos et sin multipliés par la vitesse
        df_temp = self.df.merge(df_direction, left_on=nom_colonne_dir, right_on="dir")
        df_temp[nom_colonne_dir+"_X"]=df_temp.cos*df_temp[nom_colonne_speed]
        df_temp[nom_colonne_dir+"_Y"]=df_temp.sin*df_temp[nom_colonne_speed]
        df_temp[nom_colonne_dir+"_RAD"]=df_temp["rad"]  # on garde rad pour les graphes
        
        self.df = df_temp.drop(columns=["cos", "sin", "rad", "dir", nom_colonne_dir])
    
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
        fig, axes = plt.subplots(1,1,figsize=(18,12))
        
        # correlations fortes
        # => aucune vairable n'est fortement correlee avec RainTomorrow
        #hm = sns.heatmap(abs(df_num.corr())>.5, cmap="binary", ax=axes[0])
        #axes[0].set_title("Corrélations fortes (>.5)", fontsize=18)
        
        # toutes correlations
        # => correlations intéressantes (.25<corr<.5) avec Sunshine, Humidity3pm, Humidity9am, Cloud9am (dommage...), Cloup9pm (idem), RainToday
        cmap_25 = sns.color_palette(['red', 'orange', 'yellow', 'white', 'white', 'yellow', 'orange', 'red'])
        
        dfcorr=df_num.corr()
        sns.heatmap(dfcorr, cmap=cmap_25, annot=True, fmt=".2f", ax=axes, vmin=-1, vmax=1, mask=np.triu(dfcorr))
        axes.set_title("Corrélations", fontsize=18)
        
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

        fig, axes = plt.subplots(1,1,figsize=(24,18))
        
        # affiche le nb de NA pour chaque variable et pour chaque Location
        df_nb_na = self.df.groupby('Location').apply(lambda x: x.isna().sum()).drop(columns=["Location"])
        df_nb = self.df.groupby('Location').size().reset_index(name='NbTotEnregistrements\n(nuls ou non)')
        
        df_nb_na = df_nb_na.merge(df_nb, left_index=True, right_on='Location').set_index("Location")

        self.df_nb_na=df_nb_na        
        # modifie pour avoir des %
        df_nb_na.iloc[:,:-1] = df_nb_na.iloc[:,:-1].div(df_nb_na.iloc[:,-1], axis=0)*100
        
        self.df_nb_na=df_nb_na
        
        print("\n Taux de valeurs nulles par Location pour chaque variable\n")
        print (df_nb_na.iloc[:,1:])
        
        sns.heatmap(df_nb_na.iloc[:,1:-1], cmap='gnuplot2_r', annot=True, fmt=".1f")
        axes.set_title("Taux de valeurs nulles pour chaque couple Location/variable", fontsize=18)
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
        if not hasattr(self, "df_resample"):
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
        if not hasattr(self, "df_resample"):
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
    
    # analyses de la saisonalité (in progress)
    def analyses_temporelles_saisonalite(self, colonne: str, location:str, periode:int):
        from statsmodels.tsa.seasonal import seasonal_decompose
        import statsmodels.api as sm
        
        # si l'attribut n'a pas encore été créé, alors on fait la reindexation temporelle
        if not hasattr(self, "df_resample"):
            self.reindexation_temporelle()

        #self.df_resample.loc[self.df_resample.Location==location,colonne] = self.df_resample.loc[self.df_resample.Location==location,colonne].fillna(self.df.loc[self.df.Location==location,colonne].mean())
        
        if location=="":
            # remplacement des NA par la moyenne. si autre approche (knn-imputer par exemple), il convient de l'appliquer avant cette étape
            self.df_resample.loc[:,colonne] = self.df_resample.loc[:,colonne].fillna(self.df.loc[:,colonne].mean())
            # calcule de la moyenne pour la frequence donnée (quotidienne, là) pour toutes les Location, suivi d'un dropna
            serie = self.df_resample[colonne].resample('D').mean().dropna()       
        else:
            # remplacement des NA par la moyenne. si autre approche (knn-imputer par exemple), il convient de l'appliquer avant cette étape
            self.df_resample.loc[self.df_resample.Location==location,colonne] = self.df_resample.loc[self.df_resample.Location==location,colonne].fillna(self.df.loc[self.df.Location==location,colonne].mean())
            # enleve les na (par securité, car inutile normalement vu qu'on vient de faire un fillna)
            serie = self.df_resample.loc[self.df_resample.Location==location,colonne].dropna()
        
        serie = serie["2008-12-01":"2012-12-01"]
        
        sd = seasonal_decompose(serie, period=periode)#, model='multiplicative')
        #sd.plot()
        #plt.show();
        
        fig, ax = plt.subplots(3,2,figsize=(24,12))
        cvs = serie - sd.seasonal
        cvst = cvs - sd.trend # =sd.resid

        ax = ax.reshape(-1)

        serie.plot(ax=ax[0])
        ax[0].set_title("donnnées initiales (avec saisonnalité)")

        cvs.plot(ax=ax[1])
        ax[1].set_title("sans saisonnalité")
        
        cvst.plot(ax=ax[2])
        ax[2].set_title("sans saisonnalité ni tendance (=bruit uniquement)")      
        
        sd.seasonal.plot(ax=ax[3])
        #!sd.resid.plot(ax=ax[3])
        ax[3].set_title("saisonnalité")

        sd.trend.plot(ax=ax[4])
        ax[4].set_title("tendance")

        plt.suptitle("Analyse tendance/saisonnalité/bruit de "+colonne+ " - Location: "+location+ " - Periode:"+ (str)(periode))
        plt.show();
        
        self.sd = sd
        
    # analyse la saisonnalité de toutes les variables des colonnes numeriques du DF
    def analyse_temporelle_saisonnalite_toutes_variables(self, location:str):
        coln = self.df.select_dtypes(include=['number']).columns
        
        for c in coln:
            self.analyses_temporelles_saisonalite(c, location, 365)
    
    # affiche graphe d'autocorrélation
    def analyse_temporelle_autocorr(self, colonne: str):
        from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
        from pandas.plotting import autocorrelation_plot
        plt.figure(figsize=(240,12))
        
        autocorrelation_plot(pa.df_resample.loc[pa.df_resample.Location=="Perth"][colonne].resample('D').mean().diff(1).dropna(), label=colonne)
        plt.show();
        
        fig, ax = plt.subplots(2,1,figsize=(240,12))
        plot_acf(pa.df_resample.loc[pa.df_resample.Location=="Perth"][colonne].resample('D').mean().diff(1).dropna(), lags = 1200, ax=ax[0])
        plot_pacf(pa.df_resample.loc[pa.df_resample.Location=="Perth"][colonne].resample('D').mean().diff(1).dropna(), lags = 1200, ax=ax[1])
        plt.show();
    
    # trace l'histogramme des precipitation et courbe des temperatures pour l'australie et pour chaque Location
    def analyse_annuelle_integrale(self):
        self.analyse_annuelle("")
        for lo in self.df.Location.unique():
            self.analyse_annuelle(lo)
    
    # fait une moyenne annuelle
    def analyse_annuelle(self, location:str):
        # si l'attribut n'a pas encore été créé, alors on fait la reindexation temporelle
        if not hasattr(self, "df_resample"):
            self.reindexation_temporelle()
        
        self._ajoute_colonnes_dates()
        
        # retrait des données < 1/1/2009 et >31/12/2016 pour avoir des années complètes
        df_resample = self.df_resample.loc['2009-01-01':'2016-12-31']
        
        if (location==""):
            df_filtre = df_resample.resample('D').mean().dropna()       
            df_filtre["AAMM"] = df_filtre.Annee.astype(str)+"-"+df_filtre.Mois.astype(str)
        else:
            df_filtre = df_resample[df_resample.Location == location]
        
        self.da = df_filtre.groupby('AAMM').agg({'MaxTemp': 'mean', 'Rainfall': 'sum', 'Annee':'max', 'Mois':'max'}).reset_index()
        self.da2 = self.da.groupby("Mois").agg({'MaxTemp': 'mean', 'Rainfall': 'mean'}).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        ax1.plot(self.da2['Mois'], self.da2['MaxTemp'], label='Température (°C)', color='r')
        ax1.set_yticks(np.arange(0,41,5))
        ax1.set_ylabel("Température (°C)")

        ax2 = ax1.twinx()
        ax2.bar(self.da2['Mois'], self.da2['Rainfall'], label='Précipitations (mm)', color='#06F', alpha=.5)
        ax2.set_yticks(np.arange(0,201,10))
        ax2.set_ylabel("Précipitations (mm)")

        
        nom_mois = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        plt.xticks(ticks=np.arange(len(nom_mois))+1, labels=nom_mois)
        nom_titre = 'Températures moyennes mensuelles et cumul des précipations\nAnnées 2009 à 2016'
        if (location == ""):
            nom_titre = nom_titre+" - Australie complète"
        else:
            nom_titre = nom_titre+" - "+location
        plt.title(nom_titre)
        plt.savefig("graphes\\DiagClim_"+location)
        plt.show()
        
        
    def _ajoute_colonnes_dates(self):
        self.df_resample["Date"] = self.df_resample.index
        self.df_resample["Annee"] = self.df_resample.Date.dt.year
        self.df_resample["Mois"] = self.df_resample.Date.dt.month
        self.df_resample["AAMM"] = self.df_resample.Annee.astype(str)+"-"+self.df_resample.Mois.astype(str)
        
    
    
    # verifie le periode optimale de saisonnalité
    def test_max_saisonalite(self):
        for i in range(345,370,1):
            self.analyses_temporelles_saisonalite("MaxTemp","Melbourne", i)
            print ("per ",i, 
                   "\t delta season: {:.2f}\
                    delta resid: {:.2f}\
                    std seas: {:.2f}\
                    std resid: {:.2f}"
                    .format(self.sd.seasonal.max() - self.sd.seasonal.min(),
                            self.sd.resid.max() - self.sd.resid.min(),
                            self.sd.seasonal.std(),
                            self.sd.resid.std()
                            ))
    

    # affiche repartition vent    
    def graphe_vent(self, location:str):
        if not hasattr(self.df, "WindGustDir_RAD"):       
            self.remplace_direction_vent()
        
        if (location==""):
            df_filtre = self.df
        else:
            df_filtre = self.df[self.df.Location == location]
       
        
        vc = df_filtre.WindGustDir_RAD.value_counts(normalize=True).sort_index()
        #vc.append(vc[0]) # pour fermer le tracé
        
        print (vc)
        self.vc = vc
        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        #ax.set_rlim(0,1/8)
        ax.fill(vc.index, vc.values, '#48A', alpha=.8)

        ax.set_xticks(np.arange(2*np.pi, 0, -np.pi/8))
        ax.set_xticklabels(["E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE"])
        ax.set_yticklabels([])

        nom_title="Distribution des directions du vent (WindGustDir) - "
        if (location==""):
            nom_title+="Australie complète"
        else:
            nom_title+=location
        plt.title(nom_title)
        #plt.savefig("graphes\\WinGustDir_"+location)

    # affiche le graphe de toutes les location
    def graphe_vent_integral(self):
        for lo in self.df.Location.unique():
            self.graphe_vent(lo)
        
    # -----------------------------
    # -----------------------------
    # --- gestion des NA
    # -----------------------------
    # -----------------------------
    
    # KNN imputer - (sur une seule ville pour le moment: 40mn pour lancer avec knn=1 sur tout le dataset!)
    def gestion_na_knni(self):
        
        knni = KNNImputer(n_neighbors=3)
        
        coln = self.df_resample.select_dtypes(include=['number']).columns
        print ("coln : ", coln)
        #self.df_resample_nona = pd.DataFrame(knni.fit_transform(self.df_resample), columns=self.df_resample.columns, index=self.df_resample.index)
        
        self.df_resample_nona = self.df_resample.copy()
        
        self.df_resample_nona = self.df_resample_nona[self.df_resample_nona.Location=="Melbourne"]
        
        self.df_resample_nona[coln] = pd.DataFrame(knni.fit_transform(self.df_resample_nona[coln]), columns=coln, index=self.df_resample_nona.index)
        
        fig, ax = plt.subplots(2,1, figsize=(18,12))
        ax[0].plot(self.df_resample.loc[pa.df_resample.Location=="Melbourne","MaxTemp"], label="MaxTemp de Melbourne - Données d\'origine")
        ax[0].legend()
        ax[1].plot(self.df_resample_nona.loc[pa.df_resample_nona.Location=="Melbourne","MaxTemp"], label="MaxTemp de Melbourne - Données extrapolées avec KNN Imputer à partir de Melbourne uniquement")
        ax[1].legend()
    
    # -----------------------------
    # calcule la correlation avec chaque variable qualitative du vent
    def correlation_vent(self):
        self.df_chi2=pd.DataFrame()
        var_vent=["WindGustDir", "WindDir9am", "WindDir3pm"]
        for e in var_vent:
            self._correlation_var(e)
        print("\np-value issue du Chi2 entre les variables du vent et RainTomorrow:\n",self.df_chi2)
    
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
        self.df_moyenne["Climat"] = clf.labels_[self.df_moyenne.index]
        
        self.df_climat = self.df_moyenne[["Location", "Climat"]]
        
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
#        fig = px.scatter_geo(df_rt, lat='lat', lon='lng', hover_name='Location', color='MaxTemp', size='RainTomorrow')
        fig.update_layout(mapbox_style='open-street-map')
        #fig.update_layout(margin={'r':0, 't':0, 'l':0, 'b':0})
        #fig.update_geos(projection_type='mercator')

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

#pa.analyse_donnees()
#pa.preprocessing_apres_analyse()
#pa.carte_australie()
#pa.synthetise_villes()