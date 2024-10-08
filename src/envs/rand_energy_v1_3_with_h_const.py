import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 增加能量溢出惩罚  有作用，但上限没提升
# 布朗运动

class SysEnv(gym.Env):

    def __init__(self, max_energy=100, time_interval=1, max_episode_steps=512, render_mode=None):
        self.MAX_ENERGY = np.float32(max_energy)
        self.time_interval = np.float32(time_interval)

        # Enow Ei Hi
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([self.MAX_ENERGY, self.MAX_ENERGY, 1]), dtype=np.float32)
        # 动作是消耗的能量
        self.action_space = spaces.Box(low=0, high=self.MAX_ENERGY, dtype=np.float32)
        self._max_episode_steps = max_episode_steps

    @property
    def observation(self):
        # todo random energy第一步应该是0
        return np.array([self.rest_energy, self.random_energy_arr[self._steps],
                         self.random_gain_arr[self._steps]], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # todo 用seed测试生成的序列是否一样

        self.random_energy_arr = self._random_energy_generate()
        self.random_gain_arr = self._random_gain_generate()
        self._steps = 0
        self.rest_energy = 0
        # todo 始终使用同一套数据分布测试，能做到更好吗？ 可以固定seed试试，如果不行，说明算法有问题
        return self.observation, {}

    def step(self, action):

        assert not np.isnan(action[0])
        action = action[0]

        # p_send发送功率  将action映射成真实action
        p_send = action

        upper_bound = self.observation[0]

        #todo 或许可以对于违法动作给予惩罚，降低奖励，可以考虑设置惩罚为action-upper

        # 在评估模式时，只有truncated才算结束，terminated不算结束
        # 违法超出值  excessive_amount = 0
        # 能量溢出值  buffer_overflow
        # 惩罚之和
        penalty = 0

        # 检验动作是否违法
        if p_send * self.time_interval > upper_bound:
            #能量违法溢出量
            excessive_amount = p_send * self.time_interval - upper_bound

            #惩罚剪裁
            excessive_amount = np.clip(excessive_amount, a_min=None, a_max=self.MAX_ENERGY/24,dtype=np.float32)
            # excessive_amount = np.clip(overflow, a_min=None, a_max=0.5)
            # 惩罚违法动作，但也奖励发送数据量
            # 裁剪功率到upper_bound
            p_send = upper_bound
            data = self.power_to_data(p_send)

            # todo 将溢出能量转换成惩罚，映射到与数据发送量同一量级

        else:
            data = self.power_to_data(p_send)
            excessive_amount = 0

        # print(f"self.rest_energy:{self.rest_energy}")
        # print(f"action[0]:{self.__RP_function(action[0])}")
        new_rest_energy = self.rest_energy - p_send * self.time_interval + self.random_energy_arr[self._steps]

        # 能量溢出值
        buffer_overflow = max(new_rest_energy - self.MAX_ENERGY, 0)

        # 检验能量是否溢出
        if buffer_overflow > 0:
            buffer_overflow = np.clip(buffer_overflow, a_min=None, a_max=self.MAX_ENERGY/24,dtype=np.float32)
            new_rest_energy = self.MAX_ENERGY

        penalty = np.log2(1 + (excessive_amount*1 + buffer_overflow)*0.25, dtype=np.float32)
        reward = data - penalty  # 按照D=log2 (1+p)惩罚

        #与状态有关的计算设置成float32
        # assert new_rest_energy >= 0, '精度出现错误，rest_energy<0'

        self.rest_energy = new_rest_energy
        truncated = False if self._steps + 1 < self._max_episode_steps else True

        info = {'E_now': self.rest_energy, 'E_i': self.random_energy_arr[self._steps]
            , 'c_gain': self.random_gain_arr[self._steps], 'reward': reward, 'done': truncated, 'data': data,
                'excessive_amount': excessive_amount, 'penalty': penalty}

        self._steps += 1
        return self.observation, reward, False, truncated, info

    def power_to_data(self, p):
        data = self.time_interval * np.log2(1.0 + (self.observation[2] ** 2.0) * p,dtype=np.float64)
        return np.float32(data)

    def action_wrapper(self, action):
        action = action / self.MAX_ENERGY * 2 - 1
        return np.float32(action)

    def unwrapper_action(self, action):
        action = (action + 1) * self.MAX_ENERGY / 2
        return np.float32(action)
    def _random_energy_generate(self):
        # random_energy_origin =  np.clip(np.random.normal(self.MAX_ENERGY/ 6.0, self.MAX_ENERGY / 10.0, self._max_episode_steps),
        #         a_min=0, a_max=None)
        # vec_energy_to_trans = np.vectorize(lambda x: np.log1p(x))
        # random_energy_trans = vec_energy_to_trans(random_energy_origin)
        # return random_energy_trans
        # todo 用self.np_random
        # 第一种，正态分布
        # return np.array(np.clip(self.np_random.normal(self.MAX_ENERGY / 10, self.MAX_ENERGY / 20, self._max_episode_steps+1),
        #         a_min=0, a_max=None),dtype=np.float32)
        # 第二种，均匀分布
        # 第三种，随机游走
        def brownian_motion(num_points, sigma, upper, lower=0):
            upper = np.float32(upper)
            lower = np.float32(lower)
            increments = self.np_random.normal(0, sigma, num_points - 1).astype(np.float32)
            now = np.float32(self.np_random.uniform(lower, upper))
            trajectory = [now]

            for inc in increments:
                now += inc
                if now < lower:
                    now = lower
                elif now > upper:
                    now = upper
                trajectory.append(now)
            return trajectory
            # sigma设置为upper/100

        return brownian_motion(self._max_episode_steps + 1, sigma=self.MAX_ENERGY / 20,
                               upper=self.MAX_ENERGY / 4)

    def _random_gain_generate(self):
        # 第一种 正态分布
        # return np.array(np.clip(
        #     self.np_random.normal(0.5, 0.25, self._max_episode_steps+1),
        #     a_min=0.1, a_max=1), dtype=np.float32)

        # 信道增益不变，算法是否有效
        return np.float32(0.5)*np.ones(shape=(self._max_episode_steps+1),dtype=np.float32)
        # 第二种，均匀分布
        # 第三种，随机游走





