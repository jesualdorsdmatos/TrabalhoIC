# -*- coding: utf-8 -*-

"""
Created on Wed Nov 20 15:22:25 2019

PSO
$ pip install pyswarms


Problema
Minimizar f(x) = x^2  com "global best PSO" vs "local Best PSO"

Documentação:
https://pyswarms.readthedocs.io/en/latest/
https://github.com/ljvmiranda921/pyswarms

Miranda L.J., (2018). PySwarms: a research toolkit for Particle Swarm Optimization in Python.
Journal of Open Source Software, 3(21), 433, https://doi.org/joss.00433

"""

# Import modules
import numpy as np
# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx


"""
Gobal best PSO.
"""

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
#instancia PSO GlobalBest
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)                              
#Run                                    
cost, pos = optimizer.optimize(fx.sphere, iters=1000)


"""
Local-best PSO. Necessário definir vizinhança!
"""
# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}
# Call instance of PSO LocalBest
optimizer = ps.single.LocalBestPSO(n_particles=2, dimensions=2, options=options)
# Perform optimization
cost, pos = optimizer.optimize(fx.sphere, iters=1000)


"""
Plot results
1D
"""
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
plot_cost_history(optimizer.cost_history)
plt.show()

"""
2D
"""
from pyswarms.utils.plotters.formatters import Mesher, Designer
# Plot the sphere function's mesh for better plots
m = Mesher(func=fx.sphere,limits=[(-1,1), (-1,1)])
# Adjust figure limits
d = Designer(limits=[(-1,1), (-1,1), (-0.1,1)],
             label=['x-axis', 'y-axis', 'z-axis'])

plot_contour(pos_history=optimizer.pos_history, mesher=m, designer=d, mark=(0,0))

"""
3D
"""
pos_history_3d = m.compute_history_3d(optimizer.pos_history)
animation3d = plot_surface(pos_history=pos_history_3d,mesher=m, designer=d,mark=(0,0,0))
