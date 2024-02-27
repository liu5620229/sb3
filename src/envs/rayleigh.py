import numpy as np

# 生成服从标准正态分布的两个独立随机变量

min = 1
max = 0
for i in range(100000):

    z1 = np.random.randn()
    z2 = np.random.randn()

    # 计算瑞利变量
    new = np.sqrt(z1**2 + z2**2)

    if new > max:
        max = new
    if new < min:
        min = new
print(f'min:{min} max:{max}')

