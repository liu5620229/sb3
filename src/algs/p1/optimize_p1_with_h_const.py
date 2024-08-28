import cvxpy as cp
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from envs.rand_energy_v1_3_with_h_const import SysEnv
import csv

def single_max(vec_env, index):
    print(index)
    E_0 = vec_env.get_attr("rest_energy")[index]
    E_i = vec_env.get_attr("random_energy_arr")[index]

    E_max = vec_env.get_attr("MAX_ENERGY")
    h = vec_env.get_attr("random_gain_arr")[index][:512]
    h = h**2
    E = np.insert(E_i,0,E_0)[:512]

    # 假设 N 是变量的数量，L 和 h 是已知的系数，E_max 是最大能量
    N = 512  # 示例数量
    L = 1   # 示例 L 值

    # 定义优化变量
    p = cp.Variable(N,nonneg=True)

    # 定义目标函数
    objective = cp.Maximize(cp.sum(cp.multiply(L, cp.log(1 + cp.multiply(h, p))/np.log(2))))

    # 定义约束条件
    constraints = [cp.sum(L*p[:i]) <= cp.sum(E[:i]) for i in range(1, N+1)]
    constraints += [cp.sum(E[:i]) - cp.sum(L*p[:i-1]) <= E_max for i in range(1, N+1)]

    # 定义和解决问题
    prob = cp.Problem(objective, constraints)
    prob.solve(solver='SCS',verbose=True,max_iters=1000)
    # 输出结果
    # print("最优解 p:", p.value)
    print("最优值 objective", prob.value)
    return prob.value

if __name__ == '__main__':
    data_episode=[]
    n_envs=500
    vec_env = make_vec_env(SysEnv, n_envs=n_envs, seed=2)
    vec_env.reset()
    # single_max(vec_env,2)
    for i in range(n_envs):
        data = single_max(vec_env,i)
        data_episode.append((i,data))
    with open('../csv/optimize_p1_with_h_const.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['num','data'])
        for data in data_episode:
            writer.writerow(data)

