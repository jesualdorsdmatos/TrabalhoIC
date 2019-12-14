# -*- coding: utf-8 -*-
"""
@author: Jesualdo Matos/João Neves
"""
 
 
 
########## INICIO DOS IMPORTS##########
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
import pyswarms as ps
import SwarmPackagePy as sp
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
########## FIM DOS IMPORTS##########
 



########## INICIO DO TRATAMENTO DE DADOS ########## 
tabela = pd.read_table('goog.us.txt', skiprows=1, sep = ',', names = ['Date', 'Open','High','Low','Close','Volume'],  usecols = [0,1,2,3,4,5], parse_dates = True, index_col = 0)

#shuffle para ler dados de forma aleatoria. tem que estar a false para não acontecer
train_input_data, test_input_data, train_target_data, test_target_data = train_test_split(tabela, tabela['Open'], test_size=.3, shuffle = False)

#Treino
train_input_data = np.array(train_input_data[1:])
train_target_data = np.array(train_target_data[0:(len(train_target_data)-1)])

#Teste
test_input_data = np.array(test_input_data[1:])
test_target_data = np.array(test_target_data[0:(len(test_target_data)-1)])
train_target_data = np.reshape(train_target_data,(len(train_target_data),1))
test_target_data = np.reshape(test_target_data,(len(test_target_data),1))
scaler = MinMaxScaler(feature_range=(0, 1))

 
#Normalizar
train_input_data = scaler.fit_transform(train_input_data)
train_target_data = scaler.fit_transform(train_target_data)
test_input_data = scaler.fit_transform(test_input_data)
test_target_data = scaler.fit_transform(test_target_data)
########## FIM DO TRATAMENTO DE DADOS ########## 

 



########## INICIO DA FUNCAO MLPREGRESS ##########  
def mlp_regress(hyperparameters):
    model = MLPRegressor(activation = 'logistic', hidden_layer_sizes=(150,),learning_rate_init=(hyperparameters[0]), momentum = (hyperparameters[1]))
    model.fit(train_input_data, train_target_data)
    model.predict(test_input_data)
    score = model.score(train_input_data, train_target_data)
    print(); print(model)
    loss = (1-score)
    print(loss)
    return loss
########## FIM DA FUNCAO MLPREGRESS ##########  
 


########## INICIO DA FUNCAO DE ENVIO ##########  
def f(x):
    n_particles = x.shape[0]
    for i in [range(n_particles)]:
        valor=x[i]
          
    j = mlp_regress(valor)
    return np.array(j)
########## FIM DA FUNCAO DE ENVIO ##########  


########## INICIO DA CHAMADA DA FUNCAO CAT ##########   
alh = sp.ca(10, f , 0.0001 , 0.9, 2 ,2, mr=10, smp=2,
                 spc=False, cdc=1, srd=0.1, w=0.1, c=1.05, csi=0.6)
########## FIM DA CHAMADA DA FUNCAO CAT ##########   

print ("best score :",alh.get_Gbest())
#print("h:",alh.get_agents())
mlp_regress(alh.get_Gbest())
#animation3D(alh.get_agents(), f, 10,10, sr=True)