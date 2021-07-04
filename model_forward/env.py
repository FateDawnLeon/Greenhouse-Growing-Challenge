import gym
import torch
import numpy as np

from model import Model
from constant import ENV_KEYS, EP_PATH, INIT_STATE_PATH, OUTPUT_KEYS, STATE_DICT_PATH


class GreenhouseSim(gym.Env):
    min_fw = 210
    num_control_params = 56  # count from CONTROL_KEYS
    num_env_params = len(ENV_KEYS)
    num_output_params = len(OUTPUT_KEYS)

    action_range = np.array([
        [0, 1],  # end (bool)
        # comp1.heatingpipes.pipe1.@maxTemp = 60
        # comp1.heatingpipes.pipe1.@minTemp = 0
        # comp1.heatingpipes.pipe1.@radiationInfluence = [0, 0]
        [0, 60],  # comp1.setpoints.temp.@heatingTemp
        [0, 5],  # comp1.setpoints.temp.@ventOffset
        # comp1.setpoints.temp.@radiationInfluence = [50, 150, 1]
        # @PbandVent [1] [3] needs to be descending
        [-20, 10],  # comp1.setpoints.temp.@PbandVent[0]
        [0, 30],  # comp1.setpoints.temp.@PbandVent[1]
        [11, 30],  # comp1.setpoints.temp.@PbandVent[2]
        [0, 30],  # comp1.setpoints.temp.@PbandVent[3]
        [0, 50],  # comp1.setpoints.ventilation.@startWnd
        # comp1.setpoints.ventilation.@winLeeMin = 0
        # comp1.setpoints.ventilation.@winLeeMax = 100
        # comp1.setpoints.ventilation.@winWndMin = 0
        # comp1.setpoints.ventilation.@winWndMax = 100
        [100, 200],  # common.CO2dosing.@pureCO2cap (unchangeable)
        [400, 1200],  # comp1.setpoints.CO2.@setpoint
        [400, 1200],  # comp1.setpoints.CO2.@setpIfLamps
        # @doseCapacity [1] [3] [5] needs to be descending
        [0, 33],  # comp1.setpoints.CO2.@doseCapacity[0]
        [0, 100],  # comp1.setpoints.CO2.@doseCapacity[1]
        [34, 66],  # comp1.setpoints.CO2.@doseCapacity[2]
        [0, 100],  # comp1.setpoints.CO2.@doseCapacity[3]
        [67, 100],  # comp1.setpoints.CO2.@doseCapacity[4]
        [0, 100],  # comp1.setpoints.CO2.@doseCapacity[5]
        [0, 1],  # comp1.screens.scr1.@enabled (bool, unchangeable)
        [0, 1],  # comp1.screens.scr1.@material == scr_Transparent.par (bool, unchangeable)
        [0, 1],  # comp1.screens.scr1.@material == scr_Shade.par (bool, unchangeable)
        [0, 1],  # comp1.screens.scr1.@material == scr_Blackout.par (bool, unchangeable)
        [-20, 30],  # comp1.screens.scr1.@ToutMax
        # @closeBelow [1] [3] needs to be descending
        [-20, 10],  # comp1.screens.scr1.@closeBelow[0]
        [0, 500],  # comp1.screens.scr1.@closeBelow[1]
        [11, 30],  # comp1.screens.scr1.@closeBelow[2]
        [0, 500],  # comp1.screens.scr1.@closeBelow[3]
        [501, 1500],  # comp1.screens.scr1.@closeAbove
        [0, 1],  # comp1.screens.scr1.@lightPollutionPrevention (bool)
        [0, 1],  # comp1.screens.scr2.@enabled (bool, unchangeable)
        [0, 1],  # comp1.screens.scr2.@material == scr_Transparent.par (bool, unchangeable)
        [0, 1],  # comp1.screens.scr2.@material == scr_Shade.par (bool, unchangeable)
        [0, 1],  # comp1.screens.scr2.@material == scr_Blackout.par (bool, unchangeable)
        [-20, 30],  # comp1.screens.scr2.@ToutMax
        # @closeBelow [1] [3] needs to be descending
        [-20, 10],  # comp1.screens.scr2.@closeBelow[0]
        [0, 500],  # comp1.screens.scr2.@closeBelow[1]
        [11, 30],  # comp1.screens.scr2.@closeBelow[2]
        [0, 500],  # comp1.screens.scr2.@closeBelow[3]
        [501, 1500],  # comp1.screens.scr2.@closeAbove
        [0, 1],  # comp1.screens.scr2.@lightPollutionPrevention (bool)
        [0, 1],  # comp1.illumination.lmp1.@enabled (bool)
        [0, 200],  # comp1.illumination.lmp1.@intensity (unchangeable)
        [0, 20],  # comp1.illumination.lmp1.@hoursLight (daily)
        # comp1.illumination.lmp1.@endTime = 20
        [100, 400],  # comp1.illumination.lmp1.@maxIglob
        # comp1.illumination.lmp1.@maxPARsum = 50
        [0, 100],  # crp_lettuce.Intkam.management.@plantDensity (daily)
    ])
    default_action = np.array([0, 60, 0, 0, 0, 0, 0, 50, 150, 1, 0, 0, 0, 0, 0, 0, 100, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 50,
                               0])

    bool_indices = [0, 17, 18, 19, 20, 28, 29, 30, 31, 38, 39]
    unchangeable_indices = [8, 17, 18, 19, 20, 28, 29, 30, 31, 40]
    daily_indices = [-1, -3]
    descending_indices = [(4, 6), (12, 14, 16), (23, 25), (34, 36)]
    action_parse_indices = [0, 5, 6, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 56]

    init_day_range = 20

    def __init__(self, state_dict_path=STATE_DICT_PATH, ep_path=EP_PATH):
        self.rng = np.random.default_rng()

        self.action_space = gym.spaces.Box(low=self.action_range[:, 0], high=self.action_range[:, 1])
        self.observation_space = gym.spaces.Discrete(self.num_env_params + self.num_output_params)

        self.net = Model(56 + 5 + 20, 20)
        self.net.load_state_dict(torch.load(state_dict_path))

        self.init_states = np.load(INIT_STATE_PATH)

        self.env_values = np.load(ep_path)
        self.max_step = self.env_values.shape[0]
        self.max_step = 0

        self.iter = 0
        self.state = None
        self.prev_action = None
        self.day_action = None
        self.num_spacings = 1

        self.reset()

    def parse_action(self, action):
        # record prev action
        if self.prev_action is None:
            self.prev_action = action

        # record action at the first hour at a certain day
        if self.iter % 24 == 0:
            self.day_action = action

        action[self.bool_indices] = action[self.bool_indices] > 0.5
        action[self.unchangeable_indices] = self.prev_action[self.unchangeable_indices]
        action[self.daily_indices] = self.day_action[self.daily_indices]

        # enforce descending order for some of the actions
        for t in self.descending_indices:
            val = action[t[0]]
            for idx in t[1:]:
                action[idx] = min(val, action[idx])
                val = action[idx]

        # add fixed actions
        parsed_action = self.default_action
        parsed_action[self.action_parse_indices] = action

        return parsed_action[0], parsed_action[1:]

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed=seed)

    def step(self, action: np.ndarray):
        # if exceeds the ep data, end trajectory
        if self.iter >= self.max_step:
            return self.state, 0, True, None

        # parse action to model input dim
        end, action = self.parse_action(action)

        # update spacing info
        if action[-1] != self.prev_action[-1]:
            self.num_spacings += 1

        # use nn to predict next state
        prev_state = self.state
        self.state = self.net.forward(action, self.env_values[self.iter], self.state).flatten()
        output_state = np.concatenate(self.env_values[self.iter + 1], self.state)

        # compute reward
        gain_diff = self.gain(self.state) - self.gain(prev_state)
        cost = self.fixed_cost(action) + self.variable_cost(self.env_values[self.iter])
        reward = gain_diff - cost

        # end trajectory if (action[0] a.k.a. end is True and fw > 210) or exceed max step
        done = (end and self.state[-6] > self.min_fw) or self.iter >= self.max_step

        # update step and prev_action
        self.iter += 1
        self.prev_action = action

        return output_state, reward, done, None

    def reset(self, start=None):
        # if START is none, randomly choose a start date
        if start is None:
            self.iter = self.rng.integers(0, self.init_day_range) * 24
        # otherwise, start from day START
        else:
            self.iter = start * 24

        # randomly choose a OP1 to start from
        self.state = self.init_states[self.rng.integers(0, self.init_states.shape[0])]
        self.prev_action = None
        self.day_action = None
        self.num_spacings = 1

        return np.concatenate(self.env_values[0], self.state)

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
        cost += action[8] * 0.015 / 365 / 24
        # lamp maintenance
        cost += action[40] * 0.0281 / 365 / 24
        # screen usage
        cost += (action[17] + action[28]) * 0.75 / 365 / 24
        # spacing changes
        if action[-1] != self.prev_action[-1]:
            cost += self.iter * 1.5 / 365 / 24
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
