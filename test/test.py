from envs.rand_energy_v1_3 import SysEnv

env = SysEnv()
obs = env.reset()
for i in range(env._max_episode_steps):
    print(obs)
    obs=env.step([0])