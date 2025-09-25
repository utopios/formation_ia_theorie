import numpy as np
import matplotlib.pyplot as plt
from base import ReseauSimple



# Architecture: 4 entrées -> 6 neurones cachés -> 3 sorties
reseau = ReseauSimple([4, 6, 3])

# Test du réseau
entrees = np.array([1.0, 0.5, -1.0, 2.0])
sorties = reseau.forward(entrees)
print(f"Entrées: {entrees}")
print(f"Sorties: {sorties}")

