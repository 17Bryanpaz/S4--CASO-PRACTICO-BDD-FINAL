#!/usr/bin/env python
# coding: utf-8

# In[288]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.stats.api as sms
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.compat import lzip
from sklearn.preprocessing import LabelEncoder
import warnings
import math
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# In[289]:


###PUNTO 1###
Ruta="D:/Academico/UDLA/Análitica Predictiva/Tarea 4/"
df = pd.read_csv(Ruta+'Walmart(1).csv', sep=',')
df


# In[290]:


###Punto 2###
df.describe
print(df.describe())


# In[291]:


#1 El promedio de ventas semanales de la tienda es de aproximadamente 1,046,965 unidades
#1 Esto proporciona una idea general del rendimiento de ventas de la tienda
#2 El atributo "Holiday_Flag" indica si una semana incluye un feriado (1) o no 
#(0). Puede ser interesante observar cómo las ventas varían durante las semanas de
#feriados en comparación con las semanas sin feriados
#3Los atributos "CPI" y "Unemployment" pueden proporcionar 
#información sobre el entorno económico y su posible influencia en las ventas de la tienda.


# In[292]:


###Punto 3###
df.isnull().sum()
print(df.isnull().sum())


# In[293]:


###Punto 4###
df.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
plt.show()


# In[294]:


#Las Variables que contienen datos atipicos son:
#Weekly Sales, Holiday Flag. Temperature, Unemployment


# In[295]:


#Punto 5
Q1 = df['Weekly_Sales'].quantile(0.25)
Q3 = df['Weekly_Sales'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df = df[~((df['Weekly_Sales'] < (Q1 - 1.5 * IQR)) | (df['Weekly_Sales'] > (Q3 + 1.5 * IQR)))]
print(df.shape)

sns.displot(df['Weekly_Sales'], color="blue")

Q1w = df.Weekly_Sales.quantile (0.25)
Q3w = df.Weekly_Sales.quantile (0.75)

IQRw = Q3w-Q1w #RANGO INTERCUARTILICO#
print(IQRw)

Q1h = df.Holiday_Flag.quantile (0.25)
Q3h = df.Holiday_Flag.quantile (0.75)

IQRh = Q3h-Q1h #RANGO INTERCUARTILICO#
print(IQRh)

df = df[~((df['Holiday_Flag']< (Q1h - 1.5 * IQRh)) | (df['Holiday_Flag']> (Q3h +1.5*IQRh)))]
df.shape
print(df.shape)

sns.displot(df['Holiday_Flag'], color='green')

Q1t = df.Temperature.quantile (0.25)
Q3t = df.Temperature.quantile (0.75)

IQRt = Q3t-Q1t #RANGO INTERCUARTILICO#
print(IQRt)

df = df[~((df['Temperature']< (Q1t - 1.5 * IQRt)) | (df['Temperature']> (Q3t +1.5*IQRt)))]
df.shape
print(df.shape)

sns.displot(df['Temperature'], color='yellow')

Q1u = df.Unemployment.quantile (0.25)
Q3u = df.Unemployment.quantile (0.75)

IQRu = Q3u-Q1u #RANGO INTERCUARTILICO#
print(IQRu)

df = df[~((df['Unemployment']< (Q1u - 1.5 * IQRu)) | (df['Unemployment']> (Q3u +1.5*IQRu)))]
df.shape
print(df.shape)

sns.displot(df['Unemployment'], color='skyblue')


# In[296]:


###
df


# In[297]:


###Punto 6###
df = df.drop("Holiday_Flag", axis=1)
df.corr().style.background_gradient(cmap='coolwarm')


# In[298]:


## Número de las variables
n = 8
fig = plt.figure(figsize=(12,12))
# Correlaciones en pares
corr = df.corr()
#
cols = corr.nlargest(6, "Weekly_Sales")["Weekly_Sales"].index
# Calculate correlation
for i in np.arange(1,6):
    regline = df[cols[i]]
    ax = fig.add_subplot(3,2,i)
    sns.regplot(x=regline, y=df['Weekly_Sales'], scatter_kws={"color": "royalblue", "s": 3},
                line_kws={"color": "turquoise"})
plt.tight_layout()
plt.show()
log_Weekly_Sales=np.log(df.Weekly_Sales)
df['log_Weekly_Sales']=log_Weekly_Sales


# In[299]:


###Punto 7###
#Variable dependiente: Weekly_Sales
#Variables independientes:
#Unemployment (Tasa de desempleo):Puede tener un impacto en la confianza del consumidor y en su capacidad para gastar.
#El CPI puede reflejar la inflación y el poder adquisitivo de los consumidores, lo que podría influir en las ventas.


# In[300]:


var_cuantitativas = df.select_dtypes('number').columns
var_cualitativas  =df.select_dtypes('object').columns
labelencoder = LabelEncoder()
df[var_cualitativas]=df[var_cualitativas].apply(LabelEncoder().fit_transform)


# In[301]:


###Punto 8###
#Se utilizara un modelo de regresión Lineal, debido a que las Weekly_Sales(Ventas Semanles)
#se puede explicar mediante una combinación lineal de las variables independientes.

regresion = ols("Weekly_Sales ~ CPI + Store + Temperature + Fuel_Price + Unemployment + Date", data=df)
results = regresion.fit()
print(results.summary())


# In[302]:


df2=df[df.columns.difference(['Weekly_Sales', 'log_Weekly_Sales'])]
df2


# In[303]:


df2.dtypes


# In[304]:


df2=df2.apply(pd.to_numeric)


# In[305]:


df2.dtypes


# In[306]:


vif_data = pd.DataFrame()
vif_data["feature"] = df2.columns
  
vif_data["VIF"] = [variance_inflation_factor(df2.values, i) for i in range(len(df2.columns))]

print(vif_data)


# In[307]:


regresion_2 = ols("log_Weekly_Sales ~ CPI + Store + Temperature + CPI + Unemployment + Date", data=df2)
results_2 = regresion_2.fit()


# In[308]:


print(results_2.summary())


# In[309]:


# Creamos el dataframe del VIF
vif_data = pd.DataFrame()
vif_data["feature"] = df2.columns.difference(['Fuel_Price'])
  
# Calculamos el VIF por c/variable
vif_data["VIF"] = [variance_inflation_factor(df2[df2.columns.difference(['Fuel_Price'])].values, i) \
                   for i in range(len(df2[df2.columns.difference(['Fuel_Price'])].columns))]

print(vif_data)
  


# In[310]:


# Creamos el dataframe del VIF
vif_data = pd.DataFrame()
vif_data["feature"] = df2.columns.difference(['Fuel_Price', 'Unemployment'])
  
# Calculamos el VIF por c/variable
vif_data["VIF"] = [variance_inflation_factor(df2[df2.columns.difference(['Fuel_Price', 'Unemployment'])].values, i) \
                   for i in range(len(df2[df2.columns.difference(['Fuel_Price', 'Unemployment'])].columns))]

print(vif_data)
  


# In[311]:


# Creamos el dataframe del VIF
vif_data = pd.DataFrame()
vif_data["feature"] = df2.columns.difference(['Fuel_Price', 'Unemployment','Temperature'])
  
# Calculamos el VIF por c/variable
vif_data["VIF"] = [variance_inflation_factor(df2[df2.columns.difference(['Fuel_Price', 'Unemployment','Temperature'])].values, i) \
                   for i in range(len(df2[df2.columns.difference(['Fuel_Price', 'Unemployment','Temperature'])].columns))]

print(vif_data)


# In[312]:


regresion_3 = ols("log_Weekly_Sales ~ CPI + Store + Date", data=df2)
results_3 = regresion_3.fit()


# In[313]:


print(results_3.summary())


# In[314]:


#Normalidad en los residuos
sm.qqplot(results_3.resid, line='q')


# In[315]:


import statsmodels.stats.api as sms
from statsmodels.compat import lzip
nombres = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
jarque_bera = sms.jarque_bera(results_3.resid)
lzip(nombres, jarque_bera)


# In[316]:


results_3.resid.mean()


# In[357]:


#Homocedasticidad en los residuos
print(results_3.resid)


# In[353]:


y_pred=results_3.predict()
sns.residplot(y_pred, results_3.resid)
plt.xlabel("y_pred")
plt.ylabel("residuos")
plt.title("Gráfica de residuos")


# In[ ]:




