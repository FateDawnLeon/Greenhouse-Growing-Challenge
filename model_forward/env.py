import gym
import numpy as np
import torch

from data import ENV_KEYS, OUTPUT_KEYS
from model import Model


STATE_DICT_PATH = '/home/leondawn/Greenhouse-Growing-Challenge/model_forward/exp_data/net(Linear_BN_LeakyReLU_128x3)_lr(ROP[0.001][100])_wd(1e-4)_bs(128)_ms(10000)_data(test_bo)/checkpoints/model_step=6000.pth'

EP_PATH = '/home/leondawn/Greenhouse-Growing-Challenge/collect_data/common/EP-SIM=A.npy'


class GreenhouseSim(gym.Env):
    min_fw = 210
    unchangeable_action_indices = (13, 17, 18, 23, 24, 30)
    num_control_params = 56 # count from CONTROL_KEYS
    num_env_params = len(ENV_KEYS)
    num_output_params = len(OUTPUT_KEYS)

    # TODO: set a default load path for the params
    def __init__(self, state_dict_path=STATE_DICT_PATH, ep_path=EP_PATH):
        # The spaces are NOT discrete. Here only using the # dim in the space info
        # TODO: determines space types -> Discrete or else?
        self.action_space = gym.spaces.Discrete(self.num_control_params)
        self.observation_space = gym.spaces.Discrete(self.num_env_params + self.num_output_params)

        self.net = Model(self.observation_space.n + self.action_space.n, self.observation_space.n)
        self.net.load_state_dict(torch.load(state_dict_path))

        self.env_values = np.load(ep_path)
        self.max_step = self.env_values.shape[0]

        self.step = 0
        self.state = None
        self.prev_action = None
        self.num_spacings = 1

        self.reset()

    def step(self, action: np.ndarray):
        # if exceeds the ep data, end trajectory
        if self.step >= self.max_step:
            return self.state, 0, True, None

        # record first action
        if self.prev_action is None:
            self.prev_action = action

        # correct some controls that cannot be changed
        action[self.unchangeable_action_indices] = self.prev_action[self.unchangeable_action_indices]

        # update spacing info
        if action[-1] != self.prev_action[-1]:
            self.num_spacings += 1

        # use nn to predict next state
        prev_state = self.state
        self.state = self.net.forward(action, self.env_values[self.step], self.state)

        # compute reward
        reward = self.gain(self.state) - self.gain(prev_state) - self.fixed_cost(action) \
                 - self.variable_cost(self.env_values[self.step])

        # end trajectory if (action[0] a.k.a. end is True and fw > 210) or exceed max step
        done = (action[0] and self.state[-6] > self.min_fw) or self.step >= self.max_step

        # update step and prev_action
        self.step += 1
        self.prev_action = action

        return self.state, reward, done, None

    def reset(self):
        self.step = 0
        # TODO: random initialization
        self.state = np.zeros(self.observation_space.n)
        self.prev_action = None
        self.num_spacings = 1

    def render(self, mode='human'):
        raise NotImplementedError

    @staticmethod
    def gain(state):
        fw, dmc = state[-6], state[-4]
        # mirror fresh weight
        fw = fw if fw <= 250 else 500 - fw
        if fw <= 210:
            price = 0
        elif fw <= 230:
            price = 0.4 * (fw - 210) / (230 - 210)
        elif fw <= 250:
            price = 0.4 + 0.1 * (fw - 230) / (250 - 230)
        else:
            raise ValueError
        # adjust for dmc
        if dmc < 0.045:
            price *= 0.9
        elif dmc > 0.05:
            price *= 1.1
        return price

    def fixed_cost(self, action):
        cost = 0
        # greenhouse occupation
        cost += 11.5 / 365 / 24
        # CO2 dosing capacity
        cost += action[13] * 0.015 / 365 / 24
        # lamp maintenance
        cost += action[30] * 0.0281 / 365 / 24
        # screen usage
        cost += (action[17] + action[19]) * 0.75 / 365 / 24
        # spacing changes
        if action[-1] != self.prev_action[-1]:
            cost += self.step * 1.5 / 365 / 24
        cost += self.num_spacings * 1.5 / 365 / 24
        return cost

    def variable_cost(self, ep):
        cost = 0
        # electricity cost
        if ep[-1] > 0.5:
            cost += self.state[-1] / 1000 * 0.1
        else:
            cost += self.state[-1] / 1000 * 0.06
        # heating cost
        cost += self.state[6] / 1000 * 0.03
        # CO2 cost
        cost += self.state[13] * 3600 * 0.12

        return cost
