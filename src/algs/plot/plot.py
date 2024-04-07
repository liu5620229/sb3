import matplotlib.pyplot as plt
import pandas

import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
data = pd.read_csv("output_data.csv")




# 绘图
plt.plot(data['mean_r'], label='mean_r')
plt.plot(data['mean_data'], label='mean_data')
plt.plot(data['mean_penalty'], label='mean_penalty')
plt.xlabel('Iterations')
plt.ylabel('Values')
plt.legend()
plt.show()