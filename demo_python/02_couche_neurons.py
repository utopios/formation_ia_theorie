import numpy as np
import matplotlib.pyplot as plt
from base import CoucheNeurones

# Test de la couche
couche1 = CoucheNeurones(nb_entrees=3, nb_neurones=4)
entrees = np.array([1.0, 2.0, 3.0])
sorties = couche1.forward(entrees, 'relu')
print(f"Sorties de la couche: {sorties}")
print(f"Forme des sorties: {sorties.shape}")
sorties = couche1.forward(entrees, 'sigmoid')
print(f"Sorties de la couche: {sorties}")
print(f"Forme des sorties: {sorties.shape}")
sorties = couche1.forward(entrees, 'tanh')
print(f"Sorties de la couche: {sorties}")
print(f"Forme des sorties: {sorties.shape}")