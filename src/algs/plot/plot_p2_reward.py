import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


episode_lenth = 512
# 读取CSV文件
hasty_p2 = pd.read_csv("../csv/hasty_p2.csv")
ppo_p2 = pd.read_csv("../csv/p2/ppo_data_rewards.csv")
sac_p2 = pd.read_csv("../csv/p2/sac_data_rewards.csv")
x_show_length=500
#平均
# m = hasty_p2['data'].mean()/episode_lenth
# hasty_p2 = [m]*x_show_length
# ppo_p2_data = ppo_p2['mean_data']/episode_lenth
# sac_p2_data = sac_p2['mean_data']/episode_lenth

#累计
m = hasty_p2['data'].mean()
hasty_p2 = [m]*x_show_length
ppo_p2_data = ppo_p2['mean_reward']
sac_p2_data = sac_p2['mean_reward']

# 创建图表
plt.figure(figsize=(10, 6)) # 设置图表大小

# 绘制曲线
plt.plot(np.arange(0, x_show_length, 1), ppo_p2_data, label='ppo', color='green', linestyle='--', linewidth=2)
plt.plot(np.arange(0, x_show_length, 1), sac_p2_data, label='sac', color='orange', linestyle='--', linewidth=2)

plt.rcParams['font.sans-serif'] = ['SimHei']
# 添加标题和标签
# plt.title('多个环境吞吐量的平均值随迭代次数的变化', fontsize=16) # 添加标题
plt.xlabel('迭代次数', fontsize=12) # x轴标签
plt.ylabel('平均reward', fontsize=12) # y轴标签

# 添加图例
plt.legend(loc='lower right', fontsize=10)
# 显示网格线
plt.grid(True)

# 设置坐标轴范围
plt.xlim(0, x_show_length)
plt.ylim(-700, 700)
plt.savefig('p2_reward.svg', format='svg')

# 显示图表
plt.show()



