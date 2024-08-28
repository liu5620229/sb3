import sys
sys.path.append("/")
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

import csv
# 数据

# from stable_baselines3.common.evaluation import evaluate_policy
from utils.evaluation import evaluate_policy

from envs.rand_energy_v2_0_with_h_const import SysEnv
import time


# Parallel environments

train_env = make_vec_env(SysEnv, n_envs=8, seed=0)

# Separate evaluation envs, with different parameters passed via env_kwargs
# Eval environments can be vectorized to speed up evaluation.

# monitor_dir_train = '/monitoring/train'
# train_env = VecMonitor(train_env, filename=monitor_dir_train)
#
# monitor_dir_eval = '/monitoring/eval'
# eval_env = VecMonitor(eval_env, filename=monitor_dir_eval)
n_actions = train_env.action_space.shape[0]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
policy_kwargs = dict(net_arch=[128]*2)
#初始化模型
model = DDPG(
    policy='MlpPolicy',
    env=train_env,  #使用N个环境同时训练
    learning_rate=4e-4,
    batch_size=256,  #采样数据量
    gamma=0.995,
    # learning_starts=100,
    action_noise=action_noise,
    # tau=0.012,
    # buffer_size=100000,
    # buffer_size=1000000,
    # ent_coef=,
    # train_freq=(10, 'step'),
    # gradient_steps=10,
    verbose=0,
    # policy_kwargs=policy_kwargs
)

if __name__ == '__main__':
    data_rewards = []
    n_envs=500
    for i in range(500):
        print(f'第{i}次评估{time.ctime()}')
        info = evaluate_policy(model.policy, make_vec_env(SysEnv, n_envs=n_envs, seed=2), n_eval_episodes=n_envs,
                               deterministic=True)
        mean_r = info['mean_reward']
        mean_data = info['mean_data']
        mean_penalty = info['mean_penalty']
        episode_datas = info['episode_datas']
        episode_rewards = info['episode_rewards']
        episode_penalties = info['episode_penalties']
        data_rewards.append((mean_data, mean_r))
        print(f"mean_r:{mean_r},mean_data:{mean_data}, mean_penalty:{mean_penalty}")
        print(f"episode_rewards:{episode_rewards}\n"
              f"episode_datas:{episode_datas} \n"
              f"episode_penalties:{episode_penalties}")

        print(f'第{i}次训练{time.ctime()}')
        model.learn(total_timesteps=512 * 5)

    current_time = time.strftime("%m%d_%H_%M", time.localtime())
    print(current_time + 'model_saved')
    model.save(f'models/v2/ddpg/{current_time}')
    with open('../csv/p2/ddpg_data_rewards_with_h_const.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['mean_data', 'mean_reward'])
        for data_reward in data_rewards:
            writer.writerow(data_reward)



