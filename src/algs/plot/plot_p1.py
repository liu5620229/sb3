import matplotlib.pyplot as plt
import numpy as np
import pandas

import matplotlib.pyplot as plt
import pandas as pd


episode_lenth = 512
# 读取CSV文件
hasty_p1 = pd.read_csv("../csv/hasty_p1.csv")
m = hasty_p1['data'].mean()/episode_lenth
hasty_p1 = [m]*500


optimize_p1 = pd.read_csv("../csv/optimize_p1.csv")
m = optimize_p1['data'].mean()/episode_lenth
optimize_p1 = [m]*500

ppo_p1 = pd.read_csv("../csv/p1/ppo_data_rewards.csv")
sac_p1 = pd.read_csv("../csv/p1/sac_data_rewards.csv")
ppo_p1_data = ppo_p1['mean_data']/episode_lenth
sac_p1_data = sac_p1['mean_data']/episode_lenth

# 创建图表
plt.figure(figsize=(10, 6)) # 设置图表大小

# 绘制曲线
plt.plot(np.arange(0, 500, 1), hasty_p1, label='greedy', color='blue', linestyle='--', linewidth=2)
plt.plot(np.arange(0, 500, 1), optimize_p1, label='optimal', color='red', linestyle='--', linewidth=2)
plt.plot(np.arange(0, 500, 1), ppo_p1_data, label='ppo', color='green', linestyle='--', linewidth=2)
plt.plot(np.arange(0, 500, 1), sac_p1_data, label='sac', color='orange', linestyle='--', linewidth=2)

plt.rcParams['font.sans-serif'] = ['SimHei']
# 添加标题和标签
plt.title('10个环境下的长期平均吞吐量的均值随迭代次数的变化', fontsize=16) # 添加标题
plt.xlabel('迭代次数', fontsize=12) # x轴标签
plt.ylabel('10个环境中的平均吞吐量', fontsize=12) # y轴标签

# 添加图例
plt.legend(loc='upper right', fontsize=10)

# 显示网格线
plt.grid(True)

# 设置坐标轴范围
plt.xlim(0, 500)
plt.ylim(0, 1)

# 显示图表
plt.show()



