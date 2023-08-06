# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("weatherAUS.csv")
print( df.columns)
print( df.info())
print( df.describe())
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
<<<<<<< Updated upstream
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

=======
plt.xticks(rotation=90)
>>>>>>> Stashed changes
plt.show()