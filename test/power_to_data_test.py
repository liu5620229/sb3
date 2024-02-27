import numpy as np
import matplotlib.pyplot as plt
def power_to_data(p):
    data = 1 * np.log2(1 + p)
    return data

# x = np.linspace(0, 10, 20)
# y = 1 * np.log2(1 + x)
# plt.scatter(x,y)
# plt.show()
max_energy = 20

Dmax = power_to_data(max_energy)
Di = power_to_data(max_energy/4)

print(f"Dmax Di{Dmax,Di}")

