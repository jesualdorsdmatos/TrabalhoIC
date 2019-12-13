# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:14:27 2019

Binary PSO

Feature Selection

https://pyswarms.readthedocs.io/en/latest/examples/usecases/feature_subset_selection.html

"""


# Import modules
import numpy as np
import random

# Import PySwarms
import pyswarms as ps

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


"""
toy dataset, 
10 features,
5 are informative,
5 are redundant,
2  are repeated
"""

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=15, n_classes=3,
                           n_informative=4, n_redundant=1, n_repeated=2,
                           random_state=1)

"""
Definição do classifcador 
"""
from sklearn import linear_model
classifier = linear_model.LogisticRegression()

"""
Função objetivo
"""

def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = 15
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    # Perform classification and store performance in P
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P)
        + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j


def f(x, alpha=0.88):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)


"""
Binary PSO
"""

# Initialize swarm, arbitrary
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}

# Call instance of PSO
dimensions = 15 # dimensions should be the number of features
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)

cost, pos = optimizer.optimize(f, iters=1000)

"""
Desempenho do classificador
"""
# Create two instances of LogisticRegression
classifier = linear_model.LogisticRegression()
# Get the selected features from the final positions
X_selected_features = X[:,pos==1]  # subset
# Perform classification and store performance in P
classifier.fit(X_selected_features, y)
# Compute performance
subset_performance = (classifier.predict(X_selected_features) == y).mean()
print('Subset performance: %.3f' % (subset_performance))



"""
Desempenho do classificador completo
"""
classifierDois = linear_model.LogisticRegression()
classifierDois.fit(X, y)
performance = (classifierDois.predict(X) == y).mean()
print('Complete set performance: %.3f' % (performance))

