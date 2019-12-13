# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:48:24 2019

Optimizing function with bounds


global-best PSO with the Rastrigin function bounded within [-5.12, 5.12]

"""

# Import modules
import numpy as np

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx #funções de otimização pré-definidas


# Create bounds
max_bound = 5.12 * np.ones(2)
min_bound = - max_bound
bounds = (min_bound, max_bound)

"""
What we’ll do now is to create a 10-particle, 2-dimensional swarm
"""

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO with bounds argument
optimizer = ps.single.GlobalBestPSO(n_particles=5, dimensions=2, options=options, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(fx.rastrigin, iters=100)

"""
plot optimizer performance
"""
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
plot_cost_history(optimizer.cost_history)
plt.show()