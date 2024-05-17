from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env

# from stable_baselines3.common.evaluation import evaluate_policy
from utils.evaluation import evaluate_policy

from envs.archived.rand_energy_v1_0 import SysEnv
import time


# Parallel environments

train_env = make_vec_env(SysEnv, n_envs=4, seed=0)

# Separate evaluation envs, with different parameters passed via env_kwargs
# Eval environments can be vectorized to speed up evaluation.
eval_env = make_vec_env(SysEnv, n_envs=1, seed=2)

# monitor_dir_train = '/monitoring/train'
# train_env = VecMonitor(train_env, filename=monitor_dir_train)
#
# monitor_dir_eval = '/monitoring/eval'
# eval_env = VecMonitor(eval_env, filename=monitor_dir_eval)

#初始化模型
model = TD3(
    policy='MlpPolicy',
    env=train_env,  #使用N个环境同时训练
    learning_rate=3e-4,
    learning_starts=512,
    batch_size=100,  #采样数据量
    gamma=0.995,
    train_freq=(100,'step'),
    verbose=0)

if __name__ == '__main__':
    for i in range(20):
        print(f'第{i}次评估{time.ctime()}')
        info = evaluate_policy(model.policy, make_vec_env(SysEnv, n_envs=1, seed=2), n_eval_episodes=1,
                               deterministic=True)
        mean_r: float = info['mean_reward']
        mean_data: float = info['mean_data']
        mean_penalty: float = info['mean_penalty']

        print(f"mean_r:{mean_r},mean_data:{mean_data}, mean_penalty:{mean_penalty}")

        print(f'第{i}次训练{time.ctime()}')
        model.learn(total_timesteps=512 * 50)

        current_time = time.strftime("%m%d_%H_%M", time.localtime())
        print(current_time + 'model_saved')
        model.save(f'models/td3/{current_time}')

















































