import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Exemple d'un neurone simple
class Neurone:
    def __init__(self, nb_entrees):
        self.poids = np.random.randn(nb_entrees)
        self.biais = np.random.randn()
    
    def forward(self, entrees, activation='sigmoid'):
        # Calcul de la somme pondérée
        z = np.dot(entrees, self.poids) + self.biais
        
        # Application de la fonction d'activation
        if activation == 'sigmoid':
            return sigmoid(z)
        elif activation == 'relu':
            return relu(z)
        elif activation == 'tanh':
            return tanh(z)
        else:
            return z  # linéaire


class CoucheNeurones:
    def __init__(self, nb_entrees, nb_neurones):
        # Matrice des poids (nb_entrees x nb_neurones)
        self.poids = np.random.randn(nb_entrees, nb_neurones) * 0.1
        # Vecteur des biais
        self.biais = np.zeros(nb_neurones)
    
    def forward(self, entrees, activation='sigmoid'):
        # Calcul matriciel : X * W + b
        z = np.dot(entrees, self.poids) + self.biais
        
        if activation == 'sigmoid':
            return sigmoid(z)
        elif activation == 'relu':
            return relu(z)
        elif activation == 'tanh':
            return tanh(z)
        else:
            return z
class ReseauSimple:
    def __init__(self, tailles_couches):
        self.couches = []
        
        # Créer les couches
        for i in range(len(tailles_couches) - 1):
            couche = CoucheNeurones(tailles_couches[i], tailles_couches[i + 1])
            self.couches.append(couche)
    
    def forward(self, entrees):
        activation = entrees
        
        # Propagation avant à travers toutes les couches
        for i, couche in enumerate(self.couches):
            if i == len(self.couches) - 1:  # Dernière couche (sortie)
                activation = couche.forward(activation, 'relu')
            else:  # Couches cachées
                activation = couche.forward(activation, 'tanh')
        
        return activation


import math

def calculate_tanh(x, y):
    """
    Calcule la tangente hyperbolique pour deux valeurs données.
    
    Args:
        x: Premier nombre
        y: Deuxième nombre
    
    Returns:
        Liste des valeurs tanh correspondantes [tanh(x), tanh(y)]
    """
    return [math.tanh(x), math.tanh(y)]