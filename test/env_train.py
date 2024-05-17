from envs.archived.rand_energy_v1_0 import SysEnv

from stable_baselines3 import A2C

env = SysEnv()

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1024*100)
model.set_random_seed(70)
