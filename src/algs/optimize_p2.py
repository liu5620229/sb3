import cvxpy as cp
import numpy as np

# 假设 N 是变量的数量，L 和 h 是已知的系数，E_max 是最大能量
N = 10  # 示例数量
L = 1   # 示例 L 值
h = np.array([1]+[0.1 for _ in range(N-2)]+[1])  # 示例 h 值
# h = np.ones(N)
# print(h.shape)
E_max = 10  # 示例 E_max 值
np.random.seed(0)
E = np.array([5 for i in range(N,0,-1)])# 示例 E_i 值
# E = np.array([5 for i in range(11)])

# 定义优化变量
p = cp.Variable(N)

# 定义目标函数
objective = cp.Maximize(cp.sum(cp.multiply(L, cp.log(1 + cp.multiply(h, p)))))
print(objective)

# 定义约束条件
constraints = [cp.sum(L*p[:i]) <= cp.sum(E[:i]) for i in range(1, N+1)]
constraints += [cp.sum(E[:i]) - cp.sum(L*p[:i-1]) <= E_max for i in range(1, N+1)]
print(constraints)

# 定义和解决问题
prob = cp.Problem(objective, constraints)
prob.solve(solver='ECOS',abstol=1e-10)

# 输出结果
print("最优解 p:", p.value)
