import torch
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class SysEnv(gym.Env):
    time_rate = 1
    MAX_DATA = 1
    MAX_ENERGY = 3
    def __init__(self, init_data, init_energy_trans, time_rate):
        super(SysEnv, self).__init__()
        SysEnv.time_rate = time_rate

        self.MAX_ENERGY = self.trans_to_energy(self.MAX_ENERGY_trans)
        # todo 根据能量速率关系调整maxdata 和energy
        self.rest_data = init_data
        self.rest_energy_trans = init_energy_trans
        SysEnv.time_rate = time_rate
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([self.MAX_DATA, self.MAX_ENERGY_trans]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([self.MAX_DATA]), dtype=np.float32)
        self._max_episode_steps = 1024
        self.state = np.array([self.rest_data, self.rest_energy_trans],dtype=np.float32)
        self.random_data = self.__random_arriving_data()
        self.random_energy = self.__random_arriving_energy_trans()


        self.steps = 0
        self.seed_= None

        #奖励函数中惩罚的正则化因子
        self.ratio = self.MAX_DATA / self.MAX_ENERGY
        self.lmda = 0.9
        # self.state_dim = 2
        # self.action_dim = 1
        # self.steps = 0

    def reset(self):
        self.rest_data = self.np_random.uniform(low=0, high=self.MAX_DATA)
        self.rest_energy_trans = self.np_random.uniform(low=0, high=self.MAX_ENERGY_trans)
        self.steps = 0
        #todo 始终使用同一套数据分布，能做到更好吗？
        # self.random_energy = self.__random_arriving_energy_trans()
        # self.random_data = self.__random_arriving_data()
        self.state = np.array([self.rest_data, self.rest_energy_trans],dtype=np.float32)
        return self.state

    def seed(self, seed=None):
        seed = self.seed_
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = action[0]
        #todo 是否应该把随机到来值的更新，与step步骤分开，只返回使用能量后的状态

        upper_bound = self.data_upper_bound(self.state)

        #todo 或许可以对于违法动作给予惩罚，降低奖励，可以考虑设置惩罚为action-upper
        if upper_bound - action < -0.01:
            print('动作违法，自动取action为upper_bound'
                  f'违法action:{action}'
                  f'此时的upper：{upper_bound}')
            action = upper_bound

        #todo 或许也可以对溢出的丢失数据进行惩罚
        new_rest_data = min(self.rest_data - action + self.random_data[self.steps], self.MAX_DATA)
        #精度问题,防止state为负数
        new_rest_data = max(new_rest_data,0)

        # print(f"self.rest_energy:{self.rest_energy}")
        # print(f"action[0]:{self.__RP_function(action[0])}")
        new_rest_energy_trans = min(self.rest_energy_trans - self.__RP_function_trans(action)
                              + self.random_energy[self.steps], self.MAX_ENERGY_trans)

        new_rest_energy_trans = max(new_rest_energy_trans,0)

        self.steps += 1
        done = False if self.steps < self._max_episode_steps else True

        self.rest_data = new_rest_data
        self.rest_energy_trans = new_rest_energy_trans
        self.state = np.array([self.rest_data, self.rest_energy_trans],dtype=np.float32)
        return self.state, self.__reward_function(self.state, action), done, action

    # 默认数据和能量到来默认服从正态分布，根据实际情况调整
    def __random_arriving_data(self):

        # todo 测试dat 试试均匀分布#
        return np.clip(np.random.normal(self.MAX_DATA/3.0, self.MAX_DATA/6.0, self._max_episode_steps), a_min=0, a_max=self.MAX_DATA/2)

    def __random_arriving_energy_trans(self):
        # random_energy_origin =  np.clip(np.random.normal(self.MAX_ENERGY/ 6.0, self.MAX_ENERGY / 10.0, self._max_episode_steps),
        #         a_min=0, a_max=None)
        # vec_energy_to_trans = np.vectorize(lambda x: np.log1p(x))
        # random_energy_trans = vec_energy_to_trans(random_energy_origin)
        # return random_energy_trans
        # todo 可以将
        return np.clip(np.random.normal(self.MAX_ENERGY_trans / 5.0, self.MAX_ENERGY_trans / 10.0, self._max_episode_steps), a_min=0, a_max=None)


    # 功率-速率函数（给定数据传输量）
    # para D : 数据包大小
    # return : 消耗的能量

    def __RP_function_trans(self, D):
        e = (np.power(4,D / self.time_rate)-1) * self.time_rate
        return self.energy_to_trans(e)
    def __RP_function_origin(self, D):
        e = (np.power(4,D / self.time_rate)-1) * self.time_rate
        return e

    def __reward_function(self, state, action):
        #todo 奖励函数
        return action
        # return action - self.lmda *  self.__RP_function_origin(action) * self.MAX_DATA / self.MAX_ENERGY
        # return action - 0.5 * self.__RP_function_trans(action)*self.MAX_DATA/self.MAX_ENERGY_trans


    @staticmethod
    def data_upper_bound(state):
        upper_bound = min(np.log2(SysEnv.trans_to_energy(state[1]) / SysEnv.time_rate + 1) / 2 * SysEnv.time_rate,state[0])
        return upper_bound

    @staticmethod
    def energy_to_trans(energy):
        #  log1p=log(1+p)
        trans = np.log1p(energy)
        return trans

    @staticmethod
    def trans_to_energy(trans):
        energy = np.exp(trans) - 1
        return energy



