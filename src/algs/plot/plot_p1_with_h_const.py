import matplotlib.pyplot as plt
import numpy as np
import pandas

import matplotlib.pyplot as plt
import pandas as pd


episode_lenth = 512
# 读取CSV文件
hasty_p1 = pd.read_csv("../csv/hasty_p1_with_h_const.csv")
optimize_p1 = pd.read_csv("../csv/optimize_p1_with_h_const.csv")
sac_p1 = pd.read_csv("../csv/p1/sac_data_rewards_with_h_const.csv")
x_show_length=500
# 平均
# m = hasty_p1['data'].mean()/episode_lenth
# hasty_p1 = [m]*x_show_length
#
# m = optimize_p1['data'].mean()/episode_lenth
# optimize_p1 = [m]*x_show_length
#
# ppo_p1_data = ppo_p1['mean_data']/episode_lenth
# sac_p1_data = sac_p1['mean_data']/episode_lenth

# 累计
m = hasty_p1['data'].mean()
hasty_p1 = [m]*x_show_length

m = optimize_p1['data'].mean()
optimize_p1 = [m]*x_show_length

sac_p1_data = sac_p1['mean_data']
# # 创建图表
plt.figure(figsize=(10, 6)) # 设置图表大小

# 绘制曲线
plt.plot(np.arange(0, x_show_length, 1), optimize_p1, label='optimal', color='red', linestyle='--', linewidth=2)
plt.plot(np.arange(0, x_show_length, 1), hasty_p1, label='greedy', color='blue', linestyle='--', linewidth=2)
plt.plot(np.arange(0, x_show_length, 1), sac_p1_data, label='sac', color='orange', linestyle='--', linewidth=2)

plt.rcParams['font.sans-serif'] = ['SimHei']
# 添加标题和标签
# plt.title('信道增益为常数时10个测试episode的平均总吞吐量的随迭代次数的变化', fontsize=16) # 添加标题
plt.xlabel('迭代次数', fontsize=20) # x轴标签
plt.ylabel('平均吞吐量（bits）', fontsize=20) # y轴标签
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# 添加图例
plt.legend(loc='lower right', fontsize=18)

# 显示网格线
plt.grid(True)

# 设置坐标轴范围
plt.xlim(0, x_show_length)
# plt.ylim(0, 1)
plt.savefig('p1_data_with_h_const.svg', format='svg')
# 显示图表
plt.show()



