#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[52]:


#PUNTO 1#
Ruta="D:/Academico/UDLA/Análitica Predictiva/Tarea 1/"
df = pd.read_csv(Ruta+'Dummy Data HSS(1).csv', sep=',')
df


# In[43]:


df.isnull().sum()
df
print(df.isnull().sum())
#Punto 2#
#Variable Objetivo# #sales#
#Variables Independientes# #Social Media y Radio#
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


# In[63]:


df= df.dropna()
var_cuantitativas = df.select_dtypes('number').columns
var_cualitativas  =df.select_dtypes('object').columns
df


# In[64]:


print(df.isnull().sum())
labelencoder = LabelEncoder()


# In[65]:


df[var_cualitativas] = df[var_cualitativas].apply(labelencoder.fit_transform)


# In[66]:


X = df[df.columns.difference(['Sales'])]
y = df.Sales


# In[67]:


#PUNTO 3#
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.10,random_state =123)


# In[68]:


print(X_train.shape,"",type(X_train))
print(y_train.shape,"\t ",type(y_train))
print(X_test.shape,"",type(X_test))
print(y_test.shape,"\t ",type(y_test))


# In[69]:


#PUNTO 4#
#El Modelo de Regresión Lineal por Sklearn#


# In[70]:


modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)


# In[72]:


predicciones_train = modelo_regresion.predict(X_train)
predicciones_test = modelo_regresion.predict(X_test)


# In[73]:


#Punto 5#


# In[74]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[75]:


#MSE#
MSE_train = mean_squared_error(y_train, predicciones_train)
MSE_test = mean_squared_error(y_test, predicciones_test)
print(MSE_train)
print(MSE_test)


# In[76]:


#RMSE#
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print(RMSE_train)
print(RMSE_test)


# In[78]:


#MAE#
MAE_train = mean_absolute_error(y_train, predicciones_train)
MAE_test = mean_absolute_error(y_test, predicciones_test)
print(MAE_train)
print(MAE_test)


# In[79]:


#R^2#
from sklearn.metrics import r2_score


# In[80]:


r_square_train = r2_score(y_train, predicciones_train)
r_square_test  = r2_score(y_test, predicciones_test)
print('El R^2 del subconjunto de entrenamiento es:' , r_square_train)
print('El R^2 del subconjunto de prueba es:' , r_square_test)


# In[81]:


# Print the Intercept:
print('intercepto:', modelo_regresion.intercept_)

# Print the Slope:
print('pendiente:', modelo_regresion.coef_)


# In[82]:


fig, ax = plt.subplots()
ax.plot(y_train.values)
ax.plot(predicciones_train)
plt.title("Valores observados vs. predichos en train set");


# In[83]:


fig, ax = plt.subplots()
ax.plot(y_test.values)
ax.plot(predicciones_test)
plt.title("Valores observados vs. predichos en test set");


# # PARTE 2

# In[106]:


#Punto 1
Ruta2="D:/Academico/UDLA/Análitica Predictiva/Tarea 2/"
dfp2 = pd.read_csv(Ruta2+'bank-additional-full.csv', sep=';')
dfp2


# In[105]:


#Punto 2
##La Variable Objetivo seleccionada es "y", que son las personas que si eligieron hacer sus depositos o no.
dfp2.isnull().sum()
dfp2
print(dfp2.isnull().sum())


# In[93]:


var_cuantitativas = dfp2.select_dtypes('number').columns
var_cualitativas  =dfp2.select_dtypes('object').columns
dfp2


# In[96]:


labelencoder = LabelEncoder()


# In[97]:


dfp2[var_cualitativas] = dfp2[var_cualitativas].apply(labelencoder.fit_transform)


# In[103]:


X = dfp2[dfp2.columns.difference(['y'])]
y = dfp2.y


# In[104]:


#PUNTO 3#
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.10,random_state =123)
print(X_train.shape,"",type(X_train))
print(y_train.shape,"\t ",type(y_train))S
print(X_test.shape,"",type(X_test))
print(y_test.shape,"\t ",type(y_test))


# In[ ]:


#PUNTO 4#
#El Modelo de Regresión Lineal por Sklearn#


# In[107]:


modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)


# In[108]:


predicciones_train = modelo_regresion.predict(X_train)
predicciones_test = modelo_regresion.predict(X_test)


# In[110]:


#Punto 5#
  #MSE#
MSE_train = mean_squared_error(y_train, predicciones_train)
MSE_test = mean_squared_error(y_test, predicciones_test)
print(MSE_train)
print(MSE_test)


# In[111]:


#RMSE#
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
print(RMSE_train)
print(RMSE_test)


# In[112]:


#MAE#
MAE_train = mean_absolute_error(y_train, predicciones_train)
MAE_test = mean_absolute_error(y_test, predicciones_test)
print(MAE_train)
print(MAE_test)


# In[113]:


#R^2#
r_square_train = r2_score(y_train, predicciones_train)
r_square_test  = r2_score(y_test, predicciones_test)
print('El R^2 del subconjunto de entrenamiento es:' , r_square_train)
print('El R^2 del subconjunto de prueba es:' , r_square_test)


# In[115]:


#Punto 6


# In[116]:


# Print the Intercept:
print('intercepto:', modelo_regresion.intercept_)

# Print the Slope:
print('pendiente:', modelo_regresion.coef_)

