from collections import defaultdict
import datetime
import json

import gym
import numpy as np
import torch

from model import Model
from constant import CONTROL_KEYS, ENV_KEYS, OUTPUT_KEYS, START_DATE, MATERIALS, \
    EP_PATH, INIT_STATE_PATH, STATE_DICT_PATH, NORM_DATA_PATH


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
        # choose one material 
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
        # choose one material 
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
                               0], dtype=np.float32)

    bool_indices = [0, 17, 18, 19, 20, 27, 28, 29, 30, 31, 38, 39]
    unchangeable_indices = [8, 17, 18, 19, 20, 28, 29, 30, 31, 40]
    daily_indices = [-1, -3]
    descending_indices = [[4, 6], [12, 14, 16], [23, 25], [34, 36]]
    pick_one_indices = [[18, 19, 20], [29, 30, 31]]
    action_parse_indices = [0, 5, 6, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 56]

    init_day_range = 20

    def __init__(self, state_dict_path=STATE_DICT_PATH, ep_path=EP_PATH, norm_data_path=NORM_DATA_PATH):
        self.rng = np.random.default_rng()

        self.action_space = gym.spaces.Box(low=self.action_range[:, 0], high=self.action_range[:, 1])
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(self.num_env_params + self.num_output_params,))

        self.net = Model(56 + 5 + 20, 20)
        # TODO: model is incorrect, map model outputs with OUTPUT_KEYS
        self.net.load_state_dict(torch.load(state_dict_path))

        self.init_states = np.load(INIT_STATE_PATH)  # shape: (*, 20), e.g. (15396，20)

        self.env_values = np.load(ep_path)  # shape (day*24, 5); e.g. (1680,5) 1680=70*24
        self._max_episode_steps = self.env_values.shape[0]

        self.start_iter = 0
        self.iter = 0
        self.state = None
        self.prev_action = None
        self.day_action = None
        self.num_spacings = 1

        # TODO: normalization data and function (std == 0), keep constantly with model side
        self.norm_data = self.npz2dic(norm_data_path)  # {'op_mean': op_mean, 'op_std':op_std, ...}

    # TODO: prev_action needs to projection first
    # TODO: parsed action are not satisfied, e.g. action[17]==100, should be [0/1]
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
            action[t] = -np.sort(-action[t])

        # only pick one screen matarial
        for t in self.pick_one_indices:
            action[t] = 0
            action[np.random.choice(t, 1)] = 1

        # add fixed actions
        parsed_action = self.default_action
        parsed_action[self.action_parse_indices] = action

        return parsed_action[0], parsed_action[1:], action

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed=seed)

    def step(self, action: np.ndarray):
        # # if exceeds the ep data, end trajectory
        # if self.iter >= self.max_step:
        #     return self.state, 0, True, None

        # print('action before parse:', action)
        # parse action to model input dim
        end, action, projected_action = self.parse_action(action)

        # update spacing info
        if action[-1] != self.prev_action[-1]:
            self.num_spacings += 1

        # use nn to predict next state
        # print('action after parse:', end, action)  # type: numpy.float32 unnormalize
        # print('env', self.env_values[self.iter])  # type: numpy.float64 unnormalize
        # print('op:', self.state)  # numpy.float32 normalized
        # inputs should be float32
        cp = self.normalize(action, self.norm_data['cp_mean'], self.norm_data['cp_std'])
        ep = self.normalize(self.env_values[self.iter].astype(np.float32), self.norm_data['ep_mean'],
                            self.norm_data['ep_std'])
        op_pre = self.state

        op_cur = self.net.predict_op(cp, ep, op_pre)
        op_cur = op_cur.reshape(op_cur.shape[1])
        ep_cur = self.normalize(self.env_values[self.iter + 1], self.norm_data['ep_mean'], self.norm_data['ep_std'])
        output_state = np.concatenate([ep_cur, op_cur])
        self.state = op_cur

        # TODO: caculate reward and cost with denormalize state
        # compute reward
        gain_diff = self.gain(self.denormalize(self.state, self.norm_data['op_mean'], self.norm_data['op_std']))\
             - self.gain(self.denormalize(op_pre, self.norm_data['op_mean'], self.norm_data['op_std']))
        cost = self.fixed_cost(projected_action) + self.variable_cost(self.env_values[self.iter])
        reward = gain_diff - cost

        # update step and prev_action
        self.iter += 1
        self.prev_action = action

        # end trajectory if (action[0] a.k.a. end is True and fw > 210) or exceed max step
        # TODO: sel.state should be denormalize.
        fw = self.state[4]
        done = (end and fw > self.min_fw) or self.iter >= self._max_episode_steps
        
        # TODO: std is 0, normalize to inf
        return output_state, reward, done, {'step_action': projected_action}

    def reset(self, start=None):
        # if START is none, randomly choose a start date
        if start is None:
            self.start_iter = self.rng.integers(0, self.init_day_range) * 24
        # otherwise, start from day START
        else:
            self.start_iter = start * 24
        self.iter = self.start_iter

        self.prev_action = None
        self.day_action = None
        self.num_spacings = 1

        # randomly choose a OP1 to start from
        state = self.init_states[self.rng.integers(0, self.init_states.shape[0])]
        norm_op1 = self.normalize(state, self.norm_data['op_mean'], self.norm_data['op_std'])
        norm_ep = self.normalize(self.env_values[self.iter], self.norm_data['ep_mean'], self.norm_data['ep_std'])
        output_state = np.concatenate([norm_ep, norm_op1])

        self.state = norm_op1
        
        # TODO: std is 0, normalize to inf
        return output_state

    def render(self, mode='human'):
        raise NotImplementedError

    @staticmethod
    def gain(state):
        # TODO: state should be denormalized
        fw, dmc = state[4], state[5]
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
        # TODO: action should be the projected agent action (44)
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
            cost += (self.iter - self.start_iter) * 1.5 / 365 / 24
        cost += self.num_spacings * 1.5 / 365 / 24
        return cost

    def variable_cost(self, ep):
        cost = 0
        # electricity cost
        # TODO: self.state is normalized
        if ep[-1] > 0.5:
            cost += self.state[6] / 1000 * 0.1
        else:
            cost += self.state[6] / 1000 * 0.06
        # heating cost
        cost += self.state[7] / 1000 * 0.03
        # CO2 cost
        cost += self.state[8] * 3600 * 0.12

        return cost

    @staticmethod
    def normalize(data, mean, std):
        # TODO: std=0, normaliztion has indf. mean and std should keeep constant with model
        data = (data - mean) / std
        return np.nan_to_num(data)

    @staticmethod
    def denormalize(data, mean, std):
        # TODO: std=0, normaliztion has indf. mean and std should keeep constant with model
        return data * std + mean

    @staticmethod
    def npz2dic(file):
        norm_npz = np.load(file)
        norm_data = {'cp_mean': norm_npz['cp_mean'], 'cp_std': norm_npz['cp_std'],
                    'ep_mean': norm_npz['ep_mean'], 'ep_std': norm_npz['ep_std'],
                    'op_mean': norm_npz['op_mean'], 'op_std': norm_npz['op_std']}
        return norm_data
    # map feature index to either default value or action index
    # e.g. CONTROL_KEY[i] -> action_dump_indices[i]
    # or CONTROL_KEY[i] -> actions[:, action_dump_indices[i]]
    action_dump_indices = [
        None,  # endDate
        60,
        0,
        "0 0",
        (1,),
        (2,),
        "50 150 1",
        (3, 4, 5, 6),
        (7,),
        0,
        100,
        0,
        100,
        (8,),
        (9,),
        (10,),
        (11, 12, 13, 14, 15, 16),
        (17,),
        None,  # scr1.@material, need special handling, indices are (18, 19, 20)
        (21,),
        (22, 24, 24, 25),
        (26,),
        (27,),
        (28,),
        None,  # scr2.@material, need special handling, indices are (29, 30, 31)
        (32,),
        (33, 34, 35, 36),
        (37,),
        (28,),
        (39,),
        (40,),
        (41,),
        20,
        (42,),
        50,
        (43,)
    ]

    @staticmethod
    def __day2str(day):
        return (START_DATE + datetime.timedelta(days=day)).strftime('%d-%m')

    @staticmethod
    def __get_nested_defaultdict():
        def factory():
            return defaultdict(factory)

        return defaultdict(factory)

    @staticmethod
    def __store_to_nested_dict(d, key, val):
        tokens = key.split('.')
        for token in tokens[:-1]:
            d = d[token]
        d[tokens[-1]] = val

    @staticmethod
    def dump_actions(actions: np.ndarray):
        """
        Dumps a history of actions into appropriate control json format.
        Parameters
        ----------
        actions : ndarray, shape (T, 44)
        The actions in one trajectory. T is the number of hours.
        Returns
        -------
        output_json : str
        The control json.
        """
        assert len(actions.shape) == 2
        assert actions.shape[1] == 44

        # add the control for the first hour
        actions = np.vstack((actions[np.newaxis, 0], actions))

        num_days = actions.shape[0] // 24

        # cut to integer days
        actions = actions[:num_days * 24]

        output = GreenhouseSim.__get_nested_defaultdict()
        for i, (key, target) in enumerate(zip(CONTROL_KEYS, GreenhouseSim.action_dump_indices)):
            # if from actions, TARGET is a tuple of column indices
            if isinstance(target, tuple):
                val = actions[:, target]

                # actions that needs to be casted to lower-case bool
                if target[0] in GreenhouseSim.bool_indices:
                    val = np.array([[str(bool(x)).lower() for x in lst] for lst in val])

                # type 1: unchangeable
                if target[0] in GreenhouseSim.unchangeable_indices:
                    assert val.shape[1] == 1
                    val = val[0][0]
                # type 2: daily
                elif target[0] in GreenhouseSim.daily_indices:
                    assert val.shape[1] == 1
                    # get action at the first hour of each day
                    val = val[::24, 0]
                    val = {GreenhouseSim.__day2str(day): val[day] for day in range(num_days)}
                # type 3: hourly, single value
                elif len(target) == 1:
                    val = np.reshape(val, (num_days, 24))
                    val = {
                        GreenhouseSim.__day2str(day): {
                            str(hour): val[day][hour] for hour in range(24)
                        }
                        for day in range(num_days)
                    }
                else:
                    assert len(target) % 2 == 0
                    val = np.reshape(val, (num_days, 24, len(target) // 2, 2))
                    val = {
                        GreenhouseSim.__day2str(day): {
                            str(hour): '; '.join([' '.join(map(str, x)) for x in val[day][hour]]) for hour in range(24)
                        }
                        for day in range(num_days)
                    }

            # otherwise, TARGET is the constant value that needs to be set to
            else:
                val = target

            GreenhouseSim.__store_to_nested_dict(output, key, val)

        # end date
        end_date = START_DATE + datetime.timedelta(days=num_days)
        GreenhouseSim.__store_to_nested_dict(output, 'simset.@endDate', end_date.strftime('%d-%m-%Y'))

        # materials
        GreenhouseSim.__store_to_nested_dict(output, 'comp1.screens.scr1.@material',
                                             MATERIALS[np.where(actions[0, (18, 19, 20)] == 1)[0][0]])
        GreenhouseSim.__store_to_nested_dict(output, 'comp1.screens.scr2.@material',
                                             MATERIALS[np.where(actions[0, (29, 30, 31)] == 1)[0][0]])

        return json.dumps(output, indent=4)
