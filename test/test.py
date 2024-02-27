import numpy as np


def power_to_data(p, h_i):
    # todo 或者
    data = np.log2(1.0 + (h_i ** 2.0) * p, dtype=np.float64)
    return np.float32(data)


def data_to_power(data, h_i):
    energy = (np.exp2(data, dtype=np.float64) - 1) / h_i ** 2.0
    return np.float32(energy)


p = 54.465131
data = power_to_data(p, 1.0)
p2 = data_to_power(data, 1.0)
print(data)
print(p2)
