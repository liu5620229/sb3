from envs.rand_energy_v1_0 import SysEnv

from stable_baselines3 import A2C

env = SysEnv()

model = A2C("MlpPolicy", env, verbose=1).get_vec_normalize_env()



# for i in range(100):
#     sum_reward = 0
#     data = 0
#     for i in range(1024):
#         action, _state = model.predict(obs, deterministic=True)
#         obs, reward, done, info = vec_env.step(action)
#         sum_reward+=reward[0]
#         data+=info[0]['data']
#         # VecEnv resets automatically
#         # if done:
#         #   obs = vec_env.reset()
#     print(f'sum_reward:{sum_reward}\n'
#           f'data:{data}')

