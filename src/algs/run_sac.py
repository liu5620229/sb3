from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# from stable_baselines3.common.evaluation import evaluate_policy
from utils.evaluation import evaluate_policy

from envs.rand_energy_v2_0 import SysEnv
import time


# Parallel environments

train_env = make_vec_env(SysEnv, n_envs=4, seed=0)

# Separate evaluation envs, with different parameters passed via env_kwargs
# Eval environments can be vectorized to speed up evaluation.

# monitor_dir_train = '/monitoring/train'
# train_env = VecMonitor(train_env, filename=monitor_dir_train)
#
# monitor_dir_eval = '/monitoring/eval'
# eval_env = VecMonitor(eval_env, filename=monitor_dir_eval)

#初始化模型
model = SAC(
    policy='MlpPolicy',
    env=train_env,  #使用N个环境同时训练
    learning_rate=4e-4,
    batch_size=256,  #采样数据量
    gamma=0.995,
    learning_starts=100,
    gradient_steps=60,
    # action_noise=,
    tau=0.005,
    buffer_size=100000,
    # buffer_size=1000000,
    # ent_coef=,
    train_freq=(10, 'step'),
    verbose=0)

if __name__ == '__main__':
    for i in range(50):
        print(f'第{i}次评估{time.ctime()}')
        info = evaluate_policy(model.policy, make_vec_env(SysEnv, n_envs=2, seed=2), n_eval_episodes=2,
                               deterministic=True)
        mean_r = info['mean_reward']
        mean_data = info['mean_data']
        mean_penalty = info['mean_penalty']
        episode_datas = info['episode_datas']
        episode_rewards = info['episode_rewards']
        episode_penalties = info['episode_penalties']

        print(f"mean_r:{mean_r},mean_data:{mean_data}, mean_penalty:{mean_penalty}")
        print(f"episode_rewards:{episode_rewards}\n"
              f"episode_datas:{episode_datas} \n"
              f"episode_penalties:{episode_penalties}")

        print(f'第{i}次训练{time.ctime()}')
        model.learn(total_timesteps=512 * 20)

        current_time = time.strftime("%m%d_%H_%M", time.localtime())
        print(current_time + 'model_saved')
        model.save(f'models/ppo/{current_time}')
















































