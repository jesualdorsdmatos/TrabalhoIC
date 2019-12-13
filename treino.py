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
 
########## INICIO DO LER FICHEIRO##########
df = pd.read_csv('aadr.us.txt', names=["Data","Open","High","Low","Close"], usecols=[0,1,2,3,4] , parse_dates=True, index_col=0)
########## FiM DO LER FICHEIRO##########

 
########## INICIO DO TRATAMENTO DE DADOS ########## 
treino_input, teste_input, treino_target, teste_target = train_test_split(df, df['Open'] , test_size=0.2 , shuffle=False)
 
 
treino_target = np.array(treino_target[1:])
teste_input =  np.array(teste_input[0:(len(teste_input)-1)])
treino_input = np.array(treino_input[0:(len(treino_input)-1)])
teste_target = np.array(teste_target[1:])
 
treino_target = np.reshape(treino_target,(len(treino_target),1))
 
teste_target = np.reshape(teste_target,(len(teste_target),1))
########## FIM DO TRATAMENTO DE DADOS ########## 

 
########## INICIO DE NORMALIZACAO ########## 
scaler = MinMaxScaler(feature_range=(0, 1))
treino_target = scaler.fit_transform(treino_target)
teste_input = scaler.fit_transform(teste_input)
t_tratreino_inputin = scaler.fit_transform(treino_input)
teste_target = scaler.fit_transform(teste_target)
########## FIM DE NORMALIZACAO ########## 
 
 

########## INICIO DA FUNCAO MLPREGRESS ##########  
def mlp_regress(hyperparamets):
    regress = MLPRegressor(hidden_layer_sizes=(80,) , activation='logistic', 
    solver='adam', batch_size='auto', learning_rate_init=(hyperparamets))
    regress.fit(treino_input,treino_target.ravel())
    regress.predict(teste_input)
    score = regress.score(teste_input,teste_target)
    loss = (1-score)
    print(regress)
    print(loss)
    return loss
########## FIM DA FUNCAO MLPREGRESS ##########  
 


########## INICIO DA FUNCAO DE ENVIO ##########  
def f(x):
    n_particles = x.shape[0]
    j = [mlp_regress(x[i]) for i in range(n_particles)]
    return np.array(j)
########## FIM DA FUNCAO DE ENVIO ##########  


########## INICIO DA CHAMADA DA FUNCAO CAT ##########   
alh = sp.ca(10, f , 0.2 , 1.0, 1 ,5, mr=4, smp=2,
                 spc=False, cdc=1, srd=0.1, w=0.1, c=0.5, csi=0.6)
########## FIM DA CHAMADA DA FUNCAO CAT ##########   

print ("best score :",alh.get_Gbest())
print("h:",alh.get_agents())
#animation3D(alh.get_agents(), f, 10, -10, sr=True)