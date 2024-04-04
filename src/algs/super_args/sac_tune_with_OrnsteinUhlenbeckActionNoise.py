
import sys
sys.path.append("/home/jack/lnj/sb3/src")
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
from stable_baselines3.sac import MlpPolicy

from envs.rand_energy_v2_0 import SysEnv
from utils.evaluation import evaluate_policy


def optimize_sac(trial):
    """ 学习参数优化函数 """
    # 为SAC算法设置超参数的搜索空间
    #gradient_steps = trial.suggest_int('gradient_steps', 1, 100)
    gradient_steps = 10
    #learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    learning_rate = 4e-4
    #batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    batch_size = 256
    #buffer_size = trial.suggest_categorical('buffer_size', [10**4, 10**5, 10**6, 10**7])
    #learning_starts = trial.suggest_categorical('learning_starts', [0, 100, 1000, 10000, 100000])
    #tau = trial.suggest_uniform('tau', 0.001, 0.02)
    #gamma = trial.suggest_uniform('gamma', 0.98, 0.9999)
    n_layers = trial.suggest_categorical('n_layers', [2, 3, 4, 5])
    #layer_sizes = []
    #for i in range(n_layers):
    #   layer_size = trial.suggest_categorical(f'layer_{i}_size', [32, 64, 128, 256, 512])
    #  layer_sizes.append(layer_size)
    # 创建MLP策略类
    policy = MlpPolicy
    # 通过修改policy_kwargs来指定不同的MLP结构
    policy_kwargs = dict(net_arch=[256]*n_layers)
    ent_coef = "auto"

    # 为SAC创建训练环境
    train_env = make_vec_env(SysEnv, n_envs=20, seed=0)

    # 配置与环境相适应的随机噪声
    n_actions = train_env.action_space.shape[0]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # 创建模型
    model = SAC(
        policy,
        env=train_env,
        learning_rate=learning_rate,
        #buffer_size=buffer_size,
        #learning_starts=learning_starts,
        batch_size=batch_size,
        #tau=tau,
        #gamma=gamma,
        ent_coef=ent_coef,
        action_noise=action_noise,
        gradient_steps=gradient_steps,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    model.learn(total_timesteps=512*300)


    info = evaluate_policy(model.policy, make_vec_env(SysEnv, n_envs=16, seed=2), n_eval_episodes=2, deterministic=True)
    mean_reward = info['mean_reward']
    print(f"mean_reward:{mean_reward}, mean_data{info['mean_data']}")

    return mean_reward

# 创建一个Optuna优化器实例
study = optuna.create_study(direction='maximize')
study.optimize(optimize_sac, n_trials=100,n_jobs=-1)

# 打印出最优的超参数
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
