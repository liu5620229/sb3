import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from envs.rand_energy_v1_3 import SysEnv
import csv
# 种子相同时，就算env_num变化，第0个生成的环境始终是一样的
env_num = 10
vec_env = make_vec_env(SysEnv, n_envs=env_num, seed=2)
obs = vec_env.reset()

dones = np.zeros(env_num,dtype=bool)
cum_data = np.zeros(env_num)
if __name__ == '__main__':
    while not dones.any():
        # print(f"obs:{obs}")
        obs, rewards, dones, infos = vec_env.step(obs[:,0].reshape(-1,1))
        # print(f'rewards:{rewards}')
        data = [info["data"] for info in infos]
        cum_data += data
    with open('csv/hasty_p1_with_h_const.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['num', 'data'])
        for i, data in enumerate(cum_data):
            writer.writerow((i,data))



