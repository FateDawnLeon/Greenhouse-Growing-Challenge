import os
from collections import defaultdict, OrderedDict
import datetime
import json
import random

import gym
import numpy as np
import torch

from model import ClimateModelDay, PlantModelDay
from constant import CONTROL_KEYS, ENV_KEYS, OUTPUT_KEYS, OUTPUTENV_KEYS_TO_INDEX, \
    OUTPUT_KEYS_TO_INDEX, START_DATE, MATERIALS, EP_PATH, OP_TRACES_DIR, MODEL_PATHS, CONTROL_INFO
from data import ClimateDatasetDay, PlantDatasetDay
from utils import normalize, unnormalize, get_ensemble_ckpt_paths


class GreenhouseSim(gym.Env):
    min_fw = 210
    num_cp = 56  # TODO: count this
    num_ep = len(ClimateDatasetDay.EP_KEYS) * 24
    num_op = len(ClimateDatasetDay.OP_KEYS) * 24
    num_op_in = len(PlantDatasetDay.OP_IN_KEYS) * 24
    num_pl = len(PlantDatasetDay.OP_PL_KEYS)

    action_range = np.array([
        [0, 1],         # end
        [10, 15],       # comp1.setpoints.temp.@heatingTemp - night
        [15, 30],       # comp1.setpoints.temp.@heatingTemp - day
        [0, 5],         # comp1.setpoints.temp.@ventOffset
        [0, 50],        # comp1.setpoints.ventilation.@startWnd
        [200, 800],     # comp1.setpoints.CO2.@setpoint - night
        [800, 1200],    # comp1.setpoints.CO2.@setpoint - day
        [-20, 30],      # comp1.screens.scr1.@ToutMax
        [0, 200],       # comp1.screens.scr1.@closeBelow
        [1000, 1500],   # comp1.screens.scr1.@closeAbove
        [-20, 30],      # comp1.screens.scr2.@ToutMax
        [0, 200],       # comp1.screens.scr2.@closeBelow
        [1000, 1500],   # comp1.screens.scr2.@closeAbove
        [16, 20],       # comp1.illumination.lmp1.@hoursLight
        [18, 24],       # comp1.illumination.lmp1.@endTime
        [0, 50],        # crp_lettuce.Intkam.management.@plantDensity
    ], dtype=np.float32)
    default_action = np.array([0, 60, 0, 0, 0, 0, 0, 50, 150, 1, 0, 0, 0, 0, 0, 0, 100, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 50,
                               0], dtype=np.float32)

    bool_indices = [0, 17, 18, 19, 20, 27, 28, 29, 30, 31, 38, 39]
    unchangeable_indices = [8, 17, 18, 19, 20, 28, 29, 30, 31, 40]
    daily_indices = [41, 43]
    descending_indices = [[4, 6], [12, 14, 16], [23, 25], [34, 36]]
    pick_one_indices = [[18, 19, 20], [29, 30, 31]]
    action_parse_indices = [0, 5, 6, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 56]

    init_day_range = 20

    def __init__(self, training=False, model_paths=MODEL_PATHS, full_ep_path=EP_PATH, traces_dir=TRACES_DIR):
        self.action_space = gym.spaces.Box(low=self.action_range[:, 0], high=self.action_range[:, 1])
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(self.num_ep + self.num_op + self.num_pl,))

        self.training = training

        # load initial state distribution
        # traces dir: TRACES_DIR/{IDX}/ep_trace.npy
        #             TRACES_DIR/{IDX}/op_trace.npy
        #             TRACES_DIR/{IDX}/pl_trace.npy
        self.trace_paths = [f'{traces_dir}/{f}' for f in os.listdir(traces_dir) if os.path.isdir(f)]
        self.ep_trace = None
        self.op_trace = None
        self.pl_trace = None

        # load full ep trace
        self.full_ep_trace = np.load(full_ep_path)

        # load model
        self.climate_model = ClimateModelDay(self.num_cp, self.num_ep, self.num_op, CLIMATE_NORM_DATA)
        self.climate_model.eval()
        self.plant_model = PlantModelDay(self.num_op_in, self.num_pl, PLANT_NORM_DATA)
        self.plant_model.eval()

        # state features definition
        self.ep = None
        self.op = None
        self.pl = None

        self.iter = 0
        self.trace = None
        self.op = None
        self.cp_prev = None
        self.agent_cp_daily = None
        self.num_spacings = 0  # TODO：space starts from 1 or 0?
        self.cum_head_m2 = 0
        self.learning = training

    # def parse_action(self, action):
    #     action[self.bool_indices] = action[self.bool_indices] > 0.5
    #
    #     # enforce descending order for some of the actions
    #     for t in self.descending_indices:
    #         action[t] = -np.sort(-action[t])
    #
    #     # only pick one screen material
    #     for t in self.pick_one_indices:
    #         action[t] = 0
    #         action[np.random.choice(t, 1)] = 1
    #
    #     # record prev action
    #     if self.cp_prev is None:
    #         self.cp_prev = action
    #         self.agent_cp_daily = action
    #
    #     # enforce unchangeable actions
    #     action[self.unchangeable_indices] = self.cp_prev[self.unchangeable_indices]
    #
    #     # enforce descending @plantDensity
    #     action[43] = min(action[43], self.cp_prev[43])
    #
    #     # record action at the first hour at each day
    #     if (self.iter - self.start_iter) % 24 == 0:
    #         self.agent_cp_daily = action
    #     action[self.daily_indices] = self.agent_cp_daily[self.daily_indices]
    #
    #     # add fixed actions
    #     sim_action = self.default_action
    #     sim_action[self.action_parse_indices] = action
    #
    #     # sim_action dim=57, action dim=44
    #     return sim_action[0], sim_action[1:], action

    @staticmethod
    def agent_action_to_dict(action: np.ndarray) -> OrderedDict:
        result = OrderedDict()
        idx = 0
        for k, size in CONTROL_INFO.items():
            result[k] = action[idx:idx + size]
            idx += size
        return result


    def reset(self, start_day=None):
        # randomly choose a trace
        trace_idx = np.random.choice(len(self.trace_paths))

        # EP trace: shape (num_days, 24, NUM_EP_PARAMS)
        if start_day is None:
            self.ep_trace = np.load(os.path.join(self.trace_paths[trace_idx], 'ep_trace.npy'))
        else:
            self.ep_trace = self.full_ep_trace[start_day:]
        self.ep = self.ep_trace[0]
        self.op = np.zeros(self.num_op)
        self.pl = np.zeros(self.num_pl)

        # OP trace: shape (num_days, 24, NUM_OP_PARAMS)
        self.op_trace = np.load(os.path.join(self.trace_paths[trace_idx], 'op_trace.npy'))
        # PL trace: shape (num_days, NUM_PL_PARAMS)
        self.pl_trace = np.load(os.path.join(self.trace_paths[trace_idx], 'pl_trace.npy'))

        self.iter = 0

        state = np.concatenate((self.ep, self.op, self.pl), axis=None)  # flatten
        return state

    def step(self, action: np.ndarray):
        # use model to predict next state
        # op_{d+1} = ModelClimate(cp_{d+1}, ep_{d+1}, op_{d})
        op = self.climate_model.predict(action, self.ep, self.op)
        # op^in_{d+1} = select(op_{d+1})
        op_in = op[:, PlantDatasetDay.INDEX_OP_TO_OP_IN]
        # pl_{d+1} = ModelPlant(op^in_{d+1}, pl_{d})
        pl = self.plant_model(op_in, self.pl)

        output_state = np.concatenate((self.ep_trace[self.iter + 1], op, pl))
        action_dict = self.agent_action_to_dict(action)
        reward = self.reward(action_dict, op, pl)
        done = action[0] or self.iter == len(self.ep_trace) - 1

        self.ep = self.ep_trace[self.iter + 1]
        if self.training:
            # update state to real next state in the trace
            self.op = self.op_trace[self.iter]
            self.pl = self.pl_trace[self.iter]
        else:
            # update state to predicted next state
            self.op = op
            self.pl = pl

        info = {
            # only meaningful if SELF.TRAINING = True
            'real_next_state': np.concatenate((self.ep, self.op, self.pl), axis=None)
        }

        return output_state, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError

    def gather_agent_state(self, ep, op):
        ep = normalize(ep, self.norm_data['ep_mean'], self.norm_data['ep_std'])
        op = normalize(op, self.norm_data['op_mean'], self.norm_data['op_std'])
        return np.concatenate([ep, op])

    @staticmethod
    def reward(action_dict, op, pl):
        pass

    @staticmethod
    def gain(op, it, cum_head_m2):
        fw = op[OUTPUT_KEYS_TO_INDEX["comp1.Plant.headFW"]]
        dmc = op[OUTPUT_KEYS_TO_INDEX["comp1.Plant.shootDryMatterContent"]]
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
        # adjust for density
        num_days = it // 24
        avg_head_m2 = num_days / cum_head_m2
        price *= avg_head_m2
        return price

    @staticmethod
    def fixed_cost(action, prev_action, it, num_spacings):
        # action should be the projected agent action (dim=44)
        # greenhouse occupation
        cost_occupation = 11.5 / 365 / 24
        # CO2 dosing capacity
        cost_fix_co2 = action[8] * 0.015 / 365 / 24
        # lamp maintenance
        cost_lamp = action[40] * 0.0281 / 365 / 24
        # screen usage
        cost_screen = (action[17] + action[28]) * 0.75 / 365 / 24
        # spacing changes
        if action[-1] != prev_action[-1]:
            cost_spacing = (it - 1) * 1.5 / 365 / 24
        else:
            cost_spacing = 0
        cost_spacing += num_spacings * 1.5 / 365 / 24

        cost_total = cost_occupation + cost_fix_co2 + cost_lamp + cost_screen + cost_spacing
        return cost_total, (cost_occupation, cost_fix_co2, cost_lamp, cost_screen, cost_spacing)

    @staticmethod
    def variable_cost(ep, op):
        # electricity cost
        peak_hour = ep[ENV_KEYS_TO_INDEX['common.Economics.PeakHour']]
        electricity = op[OUTPUT_KEYS_TO_INDEX['comp1.Lmp1.ElecUse']]
        if peak_hour > 0.5:
            cost_elec = electricity / 1000 * 0.1
        else:
            cost_elec = electricity / 1000 * 0.06
        # heating cost
        pipe_value = op[OUTPUT_KEYS_TO_INDEX['comp1.PConPipe1.Value']]
        cost_heating = pipe_value / 1000 * 0.03
        # CO2 cost
        pure_air_value = op[OUTPUT_KEYS_TO_INDEX['comp1.McPureAir.Value']]
        cost_var_co2 = pure_air_value * 3600 * 0.12

        cost_total = cost_elec + cost_heating + cost_var_co2

        return cost_total, (cost_elec, cost_heating, cost_var_co2)

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

        # cast to vanilla float to enforce json serializable
        actions = actions.astype(float)

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
                    val = val[::24, 0]
                    # special handling for @plantDensity
                    if target[0] == 43:
                        result = f'{1} {val[0]}'
                        for idx, v in enumerate(val[1:]):
                            if val[idx + 1] != val[idx]:
                                result += f'; {idx + 2} {val[idx + 1]}'
                        val = result
                    else:
                        # get action at the first hour of each day
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
                # type 4: hourly, multiple values
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
