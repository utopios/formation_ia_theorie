import numpy as np

fan_in, fan_out = 128, 64

# Xavier
sigma_xavier = np.sqrt(2 / (fan_in + fan_out))
print("Xavier σ =", sigma_xavier)

# He
sigma_he = np.sqrt(2 / fan_in)
print("He σ =", sigma_he)

# LeCun
sigma_lecun = np.sqrt(1 / fan_in)
print("LeCun σ =", sigma_lecun)