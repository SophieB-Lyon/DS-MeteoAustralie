# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

df = pd.read_csv("weatherAUS.csv")
#print( df.columns)
print( df.info())
#print( df.describe())
df2 = df.groupby("Location")["RainToday"].value_counts().unstack()
df2 = df2.sort_values("Yes", ascending=True)

## null/variable
nulls = df.isnull().sum()
nulls.plot(kind="bar")
plt.show()

##Grap describe
df3 = df.select_dtypes(include=['number'])
df3.boxplot()
plt.xticks(rotation=90)
plt.show()


## Grap Bars yes/no rain
print(df2)
plt.figure(figsize=(8, 8))
indices = np.arange(len(df2))
esp=0.5
plt.bar(indices, df2.No, width=esp)
plt.bar(indices+esp, df2.Yes, width=esp)
plt.xticks(indices, df2.index)
plt.legend(title='RainToday', labels=['No', 'Yes'], loc='upper right')
plt.xticks(rotation=90)
plt.show()


## Les précipitations en fonction de la différence entre les maxima et les minima. Option 2 a graph de Shopie
plt.figure(figsize=(8, 8))
df4=df.dropna(subset=["MinTemp", "MaxTemp"])
df4["diff_Temp"]=df4["MaxTemp"] - df4["MinTemp"]
df4["diff_Temp"] = df4["diff_Temp"].apply(lambda x: round(x))
count_df4 = df4.groupby(['diff_Temp', 'RainToday']).size().unstack(fill_value=0)
plt.bar(count_df4.index, count_df4.No, width=0.4)
plt.bar(count_df4.index+0.4, count_df4.Yes, width=0.4)


plt.legend(title='RainToday', labels=['No', 'Yes'], loc='upper right')
plt.xlabel('Différence de température (°C)')
plt.ylabel('Quantité de jours')
plt.title('RainDays en fonction de la différence de température')

plt.show()



### evaluer la relation entre las rain de chaque ville
df.head()
df_pivot = df.pivot(index='Date', columns='Location', values='RainToday')
df_pivot.replace({'Yes': 1, 'No': 0}, inplace=True)
#print(df_pivot)
correlation = df_pivot.corr()
correlation["Albury"]
#print("correlation", correlation)
#cmap_colors = [(-1.0, "#053061"), (-0.9, "#2166ac"), (-0.8, "#4393c3"),
               #(-0.7, "#92c5de"), (-0.6, "#fddbc7"), (0.0, "#ffffff"),
               #(0.6, "#d1e5f0"), (0.7, "#92c5de"), (0.8, "#4393c3"),
               #(0.9, "#2166ac"), (1.0, "#053061")]

# Crear la paleta de colores personalizada
cmap = sns.color_palette("coolwarm", as_cmap=True, n_colors=20)

plt.figure(figsize=(25, 25))
sns.heatmap(correlation, annot=True, cmap=cmap, center=0, linewidths=.5, fmt=".1f", vmin=-1, vmax=1)
plt

##Nulos por diudad y fecha
grouped = df.groupby('Location')

for city, group in grouped:
    plt.figure(figsize=(8, 5))
    null_counts = group.isnull().sum()
    null_dates = null_counts[null_counts > 0].index
    plt.bar(null_dates, null_counts[null_dates], color=sns.color_palette('pastel'))
    plt.ylim(0, 3040) ##Igual escala
    plt.title(f'Valores Nulos en {city}')
    plt.xlabel('Variables')
    plt.ylabel('Cantidad de Valores Nulos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



## Calcul correlatoin par variable entre Villes




def ciudades_mas_correlacionadas(dataframe, ville, variable_objetivo, num_vecinos):
    # Filtrar el DataFrame auxiliar para incluir solo las variables indicadas para cada ciudad
    aux = dataframe.pivot(index='Date', columns='Location', values=variable_objetivo)
    print(aux)
    # Calcular la matriz de correlaciones usando la función corr()
    correlacion_matrix = aux.corr()
    
    # Obtener las ciudades más correlacionadas para la ciudad objetivo
    ciudades_correlacionadas = correlacion_matrix[ville].sort_values(ascending=False)[1:num_vecinos+1]
    
    # Crear un DataFrame con los resultados
    resultados = pd.DataFrame({'Ciudad_Correlacionada': ciudades_correlacionadas.index,
                                'Correlacion': ciudades_correlacionadas.values})
    
    return resultados

print( ciudades_mas_correlacionadas(df, "Williamtown", "Evaporation", 6) )

