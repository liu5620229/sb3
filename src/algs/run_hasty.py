import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from envs.rand_energy_v1_0 import SysEnv
# 种子相同时，就算env_num变化，第0个生成的环境始终是一样的
env_num = 1
vec_env = make_vec_env(SysEnv, n_envs=env_num, seed=2)
obs = vec_env.reset()

dones = np.zeros(env_num,dtype=bool)
cum_rewards=np.zeros(env_num)
while not dones.any():
    print(f"obs:{obs}")
    obs, rewards, dones, infos = vec_env.step(obs[:,0].reshape(-1,1))
    print(f'rewards:{rewards}')
    print(f'data:{[info["data"] for info in infos]}')
    cum_rewards += rewards
print(cum_rewards)
print(np.mean(cum_rewards))


