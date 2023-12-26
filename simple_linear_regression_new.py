# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:13:45 2023

@author: ivoto
"""

#Regresion Linal Simple 
#Importamo librerias
import numpy as np #Para trabajar con math
import matplotlib.pyplot as plt # Para la vizualizacion de datos 
import pandas as pd #para la carga de datos 

#Importamos el dataSet 

dataset = pd.read_csv('Salary_Data.csv')
#Definimos variable independientes, [filas, columnas (:-1 por que quiero todas menos la ultima)]
X = dataset.iloc[:, :-1].values 
#Definimos y(minuscula ya que es un vector en vez de una matriz) variable dependiente 
y = dataset.iloc[:, 1].values 

#Training & Test 

#Dividir el dataset en conjunto de entrenamiento y de testing
#Utilizamos un libreria sklearn, model_selection (muy utilizidad para cross-validation)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0 )

#No es necesario en la Regresion Lineal Simple, no es necesario Escalar(Normalizar)

#Escalado de variables

#Como la distancia euclidea (pitagoras de las variables ) puede tomar datos muy grande de una de las variables, lo que se hace es normalizar 
#Esto se hace para que un variable de valor muy grande no domine sobre el resto
#Escalar los datos = Normalizar los datos (Escalar a 0 y -1 correspode que el val max es 1 y el valor min es -1 ) 
#Hay dos metodos Standardisation(Campara de Gauss) y Normalisation(0 a 1 Lineal)
from sklearn.preprocessing import StandardScaler
#Escalamos el conjunto de entrenamiento 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #Ahora quedan escalados entre -1 y 1 pero es una STANDARIZACION (Normal) por lo que tendremos valores mayores a 1 y menores a -1
#El conjunto de test tiene que escalar con la misma tranformacion, no podemos usar una distinta para el conj de test 
X_test = sc_X.transform(X_test) #Solo detecta la transformacion y la aplica
#Ahora las variables de y-train e y_test no lo hacemos ya que es de clasificaion, por lo que no normalizamos 
#Si utilizaramos un algoritmo de prediccion(regresion lineal) hay que normalizar la y_train



#Crear modelo de Regresion Lineal Simple 

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de test 

#Aca le pasamos los datos de test(X_test) y comparamo el resultado con y_test para que ver que tal lo ha hecho el modelo de regresion lineal
y_pred = regression.predict(X_test)


#Visualizar los resultados de entrenamiento

#Realizamos lo nube de dispercion para comprar los resultado de la prediccion con los valores obtenidos 
plt.scatter(X_train, y_train, color = 'red') #Pintamos puntos de los valores reales 
#Trazamos la recta de regresion
plt.plot(X_train, regression.predict(X_train), color = 'blue') #Pintamos la recta de regresion que muestra los valores predichos 
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel('Sueldo en USD')
plt.show()

#Visualizar los resultados de test

plt.scatter(X_test, y_test, color = 'red')  
#Importante la recta de regresion, es la misma, ya que se tiene informacion establecida, y no depende si la pinto con el conjunto de test o de entrenamiento 
#La recta de regresion es unica 
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel('Sueldo en USD')
plt.show()













































































