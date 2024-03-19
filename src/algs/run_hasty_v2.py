import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from envs.rand_energy_v2_0 import SysEnv
# 种子相同时，就算env_num变化，第0个生成的环境始终是一样的
env_num = 1
vec_env = make_vec_env(SysEnv, n_envs=env_num, seed=2)
obs = vec_env.reset()

dones = np.zeros(env_num,dtype=bool)
cum_rewards=np.zeros(env_num)

def power_to_data(p, h_i):
    # todo 或者
    data = np.log2(1.0 + (h_i ** 2.0) * p, dtype=np.float64)
    return np.float32(data)

def data_to_power(data, h_i):
    energy = (np.exp2(data, dtype=np.float64) - 1) / h_i ** 2.0
    return np.float32(energy)

while not dones.any():
    print(f"obs:{obs}")
    # action_arr = []
    # for obs in obs:
    #     rest_energy = obs[0]
    #     rest_data = obs[3]
    #     rest_gain = obs[2]
    #     rest_data_to_energy = data_to_power(rest_data, rest_gain)
    #     action = min(rest_energy, rest_data_to_energy)
    #     action_arr.append([action])
    #
    # print(f"actions:{action_arr}")
    obs, rewards, dones, infos = vec_env.step(obs[:,0].reshape(-1,1))
    print(f'rewards:{rewards}')
    print(f'data:{[info["data"] for info in infos]}')
    cum_rewards += [info["data"] for info in infos]
print(cum_rewards)
print(np.mean(cum_rewards))


