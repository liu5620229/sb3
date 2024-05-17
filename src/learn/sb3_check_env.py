from stable_baselines3.common.env_checker import check_env
from envs.archived.rand_energy_v1_0 import SysEnv

env = SysEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)

