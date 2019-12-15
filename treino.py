# -*- coding: utf-8 -*-
"""
@author: Jesualdo Matos/João Neves
"""
 
 
 
################################ INICIO DOS IMPORTS ################################
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
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from matplotlib import animation, rc
from IPython.display import HTML
################################ FIM DOS IMPORTS ################################
 



################################ INICIO DO TRATAMENTO DE DADOS ################################
tabela = pd.read_table('goog.us.txt', skiprows=1, sep = ',', names = ['Date', 'Open','High','Low','Close','Volume'],  usecols = [0,1,2,3,4,5], parse_dates = True, index_col = 0)

treino_input, teste_input, treino_target, teste_target = train_test_split(tabela, tabela['Open'], test_size=.3, shuffle = False)


treino_input = np.array(treino_input[0:(len(treino_input)-1)])
treino_target = np.array(treino_target[1:])

teste_input = np.array(teste_input[0:(len(teste_input)-1)])
teste_target = np.array(teste_target[1:])

treino_target = np.reshape(treino_target,(len(treino_target),1))
teste_target = np.reshape(teste_target,(len(teste_target),1))

scaler = MinMaxScaler(feature_range=(0, 1))

treino_input = scaler.fit_transform(treino_input)
treino_target = scaler.fit_transform(treino_target)
teste_input = scaler.fit_transform(teste_input)
teste_target = scaler.fit_transform(teste_target)
################################ FIM DO TRATAMENTO DE DADOS ################################ 

 



################################ INICIO DA FUNCAO MLPREGRESS ################################  
def mlp_regress(hyperparameters):
    model = MLPRegressor(activation = 'logistic', hidden_layer_sizes=(150,),learning_rate_init=(hyperparameters[0]), momentum = (hyperparameters[1]))
    model.fit(treino_input, treino_target)
    model.predict(teste_input)
    score = model.score(treino_input, treino_target)
    print(); print(model)
    loss = (1-score)
    print(loss)
    return loss
################################ FIM DA FUNCAO MLPREGRESS ################################  
 


################################ INICIO DA FUNCAO DE ENVIO ################################  
def f(x):
    n_particles = x.shape[0]
    for i in [range(n_particles)]:
        valor=x[i]
          
    j = mlp_regress(valor)
    return np.array(j)
################################ FIM DA FUNCAO DE ENVIO ################################  


################################ INICIO DA CHAMADA DA FUNCAO CAT ################################   
alh = sp.ca(10, f , 0.0001 , 0.9, 2 ,2, mr=10, smp=2, spc=False, cdc=1, srd=0.1, w=0.1, c=1.05, csi=0.6)
################################ FIM DA CHAMADA DA FUNCAO CAT ################################   


################################ INICIO PRINT RESULTADO ################################
print ("Melhor Score :",alh.get_Gbest()) 
################################ FIM PRINT RESULTADO ################################


################################ INICIO GRAFICO ################################
rc('animation', html='html5')
# Initialize mesher with sphere function
m = Mesher(func=fx.sphere)

 

animation = plot_contour(pos_history=alh.get_agents(),
                         mesher=m,
                         mark=(0,0))

 

HTML(animation.to_html5_video())

################################ FIM GRAFICO ################################