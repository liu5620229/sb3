import sys
sys.path.append("/")
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.evaluation import evaluate_policy
from utils.evaluation import evaluate_policy
from envs.rand_energy_v2_0 import SysEnv
import time
import csv


# Parallel environments

train_env = make_vec_env(SysEnv, n_envs=8, seed=0)

# Separate evaluation envs, with different parameters passed via env_kwargs
# Eval environments can be vectorized to speed up evaluation.

# monitor_dir_train = '/monitoring/train'
# train_env = VecMonitor(train_env, filename=monitor_dir_train)
#
# monitor_dir_eval = '/monitoring/eval'
# eval_env = VecMonitor(eval_env, filename=monitor_dir_eval)

#初始化模型
model = PPO(
    policy='MlpPolicy',
    env=train_env,  #使用N个环境同时训练
    learning_rate=3e-4,
    # buffer_size=ff,
    n_steps=512,  #运行N步后执行更新,buffer_size=n_steps*环境数量
    batch_size=128,  #采样数据量
    n_epochs=16,  #每次采样后训练的次数
    gamma=0.995,
    verbose=0)

if __name__ == '__main__':
    data_rewards = []
    n_envs = 10
    for i in range(500):
        print(f'第{i}次评估{time.ctime()}')
        info = evaluate_policy(model.policy, make_vec_env(SysEnv, n_envs=n_envs, seed=2), n_eval_episodes=n_envs, deterministic=True)
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
    model.save(f'models/v2/ppo/{current_time}')
    with open('../csv/p2/ppo_data_rewards.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['mean_data', 'mean_reward'])
        for data_reward in data_rewards:
            writer.writerow(data_reward)

