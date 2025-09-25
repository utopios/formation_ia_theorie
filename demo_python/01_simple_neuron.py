import numpy as np
import matplotlib.pyplot as plt
from base import Neurone, sigmoid, relu, tanh




# Test du neurone
neurone = Neurone(3)
entrees = np.array([1.0, 2.0, 3.0])
sortie = neurone.forward(entrees, 'sigmoid')
print(f"Sortie du neurone: {sortie}")
sortie = neurone.forward(entrees, 'relu')
print(f"Sortie du neurone: {sortie}")
sortie = neurone.forward(entrees, 'tanh')
print(f"Sortie du neurone: {sortie}")

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid(True)

plt.tight_layout()
plt.show()