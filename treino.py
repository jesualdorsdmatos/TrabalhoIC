# -*- coding: utf-8 -*-
"""
@author: Jesualdo Matos/João Neves
"""
 
"""
        :param n: number of agents
        :param function: test function
        :param lb: lower limits for plot axes
        :param ub: upper limits for plot axes
        :param dimension: space dimension
        :param iteration: number of iterations
        :param mr: number of cats that hunt (default value is 10)
        :param smp: seeking memory pool (default value is 2)
        :param spc: self-position considering (default value is False)
        :param cdc: counts of dimension to change (default value is 1)
        :param srd: seeking range of the selected dimension
        (default value is 0.1)
        :param w: constant (default value is 0.1)
        :param c: constant (default value is 1.05)
        :param csi: constant (default value is 0.6)
"""
"""
link do MLPRegressor:https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
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
    model = MLPRegressor(activation = 'logistic', solver = 'adam',max_iter=500,hidden_layer_sizes=(200,),learning_rate_init=(hyperparameters[0]), beta_1 = (hyperparameters[1]),beta_2 = (hyperparameters[2]))
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
alh = sp.ca(10, f , 0.001 , 0.8, 3 ,30, mr=5, smp=2, spc=False, cdc=1, srd=0.01, w=0.1, c=1.05, csi=0.6)
################################ FIM DA CHAMADA DA FUNCAO CAT ################################   


################################ INICIO PRINT RESULTADO ################################
print ("Melhor Score :",alh.get_Gbest()) 
################################ FIM PRINT RESULTADO ################################








################################ INICIO GRAFICO DE COMPARACAO ################################
model = MLPRegressor(activation = 'logistic',max_iter=500, hidden_layer_sizes=(200,),solver = 'adam',learning_rate_init=(alh.get_Gbest()[0]), beta_1 = (alh.get_Gbest()[1]),beta_2 = (alh.get_Gbest()[2]))
model.fit(treino_input, treino_target.ravel())
previsao = model.predict(teste_input)
score = model.score(teste_input, teste_target)
loss = (1-score)
print("loss",loss)


plt.figure(figsize=(16,8))
plt.plot(teste_target, label='Preco de abertura esperado', color='red')
plt.plot(previsao, label='Preco de abertura previsto', color='blue')
plt.title("Treino e Teste")
plt.ylabel('Valor de abertura')
plt.xlabel('DIAS')
plt.legend()
plt.show()
################################ FIM GRAFICO DE COMPARACAO ################################






################################ INICIO ANIMACAO 3D ################################
rc('animation', html='html5')
 #Initialize mesher with sphere function
m = Mesher(func=fx.sphere)

# Obtain a position-fitness matrix using the Mesher.compute_history_3d()
# method. It requires a cost history obtainable from the optimizer class
#pos_history_3d = m.compute_history_3d(alh.get_agents())

# Make a designer and set the x,y,z limits to (-1,1), (-1,1) and (-0.1,1) respectively
from pyswarms.utils.plotters.formatters import Designer
d = Designer(limits=[(-1,1), (-1,1), (-0.1,1)], label=['x-axis', 'y-axis', 'z-axis'])

# Make animation
animation3d = plot_surface(pos_history=np.array(alh.get_agents()), # Use the cost_history we computed
                           mesher=m, designer=d,       # Customizations
                           mark=(0,0,0))               # Mark minima

# Enables us to view it in a Jupyter notebook
HTML(animation3d.to_html5_video())

################################ FIM ANIMMACAO 3D ################################
