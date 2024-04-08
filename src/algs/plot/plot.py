import matplotlib.pyplot as plt
import pandas

import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
hasty_p1 = pd.read_csv("../csv/hasty_p1.csv")
m = hasty_p1['data'].mean()
x_hasty_p1 = [m]*10

hasty_p2 = pd.read_csv("../csv/hasty_p2.csv")
m = hasty_p2['data'].mean()
x_hasty_p2 = [m]*10

optimize_p1 = pd.read_csv("../csv/optimize_p1.csv")
m = optimize_p1['data'].mean()
x_optimize_p1 = [m]*10
print(x_optimize_p1)









# # 绘图
# plt.plot(data['mean_r'], label='mean_r')
# plt.plot(data['mean_data'], label='mean_data')
# plt.plot(data['mean_penalty'], label='mean_penalty')
# plt.xlabel('Iterations')
# plt.ylabel('Values')
# plt.legend()
# plt.show()