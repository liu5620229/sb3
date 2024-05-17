import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from envs.archived.rand_energy_v1_0 import SysEnv

vec_env = make_vec_env(SysEnv, n_envs=1, seed=2)
vec_env.reset()
e_arr = vec_env.get_attr("random_energy_arr")[0]

print(np.insert(e_arr,0,0)[:513])
print(np.insert(e_arr,0,0)[:512].shape)
