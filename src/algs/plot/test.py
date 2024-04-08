import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# 创建图表
plt.figure(figsize=(10, 6)) # 设置图表大小

# 绘制曲线
plt.plot(x, y1, label='sin(x)', color='blue', linestyle='-', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='red', linestyle='--', linewidth=2)
plt.plot(x, y3, label='tan(x)', color='green', linestyle=':', linewidth=2)

# 添加标题和标签
plt.title('Trigonometric Functions Comparison', fontsize=16) # 添加标题
plt.xlabel('x', fontsize=12) # x轴标签
plt.ylabel('y', fontsize=12) # y轴标签

# 添加图例
plt.legend(loc='upper right', fontsize=10)

# 显示网格线
plt.grid(True)

# 设置坐标轴范围
plt.xlim(0, 10)
plt.ylim(-2, 2)

# 显示图表
plt.show()