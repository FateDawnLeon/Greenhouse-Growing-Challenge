from collections import defaultdict
import datetime
import json
import random

import gym
import numpy as np
import torch

from model import AGCModelEnsemble
from constant import CONTROL_KEYS, ENV_KEYS, OUTPUT_KEYS, ENV_KEYS_TO_INDEX, \
                                    OUTPUT_KEYS_TO_INDEX, START_DATE, MATERIALS, EP_PATH, OP_TRACES_PATH,\
                                    MODEL_PATHS
from utils import normalize, unnormalize, get_ensemble_ckpt_paths

class GreenhouseSim(gym.Env):
    min_fw = 210
    num_control_params = 56  # count from CONTROL_KEYS
    num_env_params = len(ENV_KEYS)
    num_op_params= len(OUTPUT_KEYS)

    action_range = np.array([
        [0, 1],  # end (bool); sim_idx: 0; agent_idx: 0  
        # comp1.heatingpipes.pipe1.@maxTemp = 60; sim_idx: 1
        # comp1.heatingpipes.pipe1.@minTemp = 0; sim_idx: 2
        # comp1.heatingpipes.pipe1.@radiationInfluence = [0, 0]; sim_idx: 3,4
        [0, 40],  # comp1.setpoints.temp.@heatingTemp; sim_idx: 5; agent_idx: 1
        [0, 5],  # comp1.setpoints.temp.@ventOffset; sim_idx: 6; agent_idx: 2
        # comp1.setpoints.temp.@radiationInfluence = [50, 150, 1]; sim_idx: 7, 8, 9
        # @PbandVent [1] [3] needs to be descending
        [-20, 10],  # comp1.setpoints.temp.@PbandVent[0]; sim_idx: 10; agent_idx: 3
        [1, 30],  # comp1.setpoints.temp.@PbandVent[1]; sim_idx: 11; agent_idx: 4
        [11, 30],  # comp1.setpoints.temp.@PbandVent[2]; sim_idx: 12; agent_idx: 5
        [1, 30],  # comp1.setpoints.temp.@PbandVent[3]; sim_idx: 13; agent_idx: 6
        [0, 50],  # comp1.setpoints.ventilation.@startWnd; sim_idx: 14; agent_idx: 7
        # comp1.setpoints.ventilation.@winLeeMin = 0; sim_idx: 15
        # comp1.setpoints.ventilation.@winLeeMax = 100; sim_idx: 16
        # comp1.setpoints.ventilation.@winWndMin = 0; sim_idx: 17
        # comp1.setpoints.ventilation.@winWndMax = 100; sim_idx: 18
        [100, 200],  # common.CO2dosing.@pureCO2cap (unchangeable); sim_idx: 19; agent_idx: 8
        [400, 1200],  # comp1.setpoints.CO2.@setpoint; sim_idx: 20; agent_idx: 9
        [400, 1200],  # comp1.setpoints.CO2.@setpIfLamps; sim_idx: 21; agent_idx: 10
        # @doseCapacity [1] [3] [5] needs to be descending
        [0, 33],  # comp1.setpoints.CO2.@doseCapacity[0]; sim_idx: 22; agent_idx: 11
        [0, 100],  # comp1.setpoints.CO2.@doseCapacity[1]; sim_idx: 23; agent_idx: 12
        [34, 66],  # comp1.setpoints.CO2.@doseCapacity[2]; sim_idx: 24; agent_idx: 13
        [0, 100],  # comp1.setpoints.CO2.@doseCapacity[3]; sim_idx: 25; agent_idx: 14
        [67, 100],  # comp1.setpoints.CO2.@doseCapacity[4]; sim_idx: 26; agent_idx: 15
        [0, 100],  # comp1.setpoints.CO2.@doseCapacity[5]; sim_idx: 27; agent_idx: 16
        [0, 1],  # comp1.screens.scr1.@enabled (bool, unchangeable); sim_idx: 28; agent_idx: 17
        # choose one material 
        [0, 1],  # comp1.screens.scr1.@material == scr_Transparent.par (bool, unchangeable); sim_idx: 29; agent_idx: 18
        [0, 1],  # comp1.screens.scr1.@material == scr_Shade.par (bool, unchangeable); sim_idx: 30; agent_idx: 19
        [0, 1],  # comp1.screens.scr1.@material == scr_Blackout.par (bool, unchangeable); sim_idx: 31; agent_idx: 20
        [-20, 30],  # comp1.screens.scr1.@ToutMax; sim_idx: 32; agent_idx: 21
        # @closeBelow [1] [3] needs to be descending
        [-20, 10],  # comp1.screens.scr1.@closeBelow[0]; sim_idx: 33; agent_idx: 22
        [0, 500],  # comp1.screens.scr1.@closeBelow[1]; sim_idx: 34; agent_idx: 23
        [11, 30],  # comp1.screens.scr1.@closeBelow[2]; sim_idx: 35; agent_idx: 24
        [0, 500],  # comp1.screens.scr1.@closeBelow[3]; sim_idx: 36; agent_idx: 25
        [501, 1500],  # comp1.screens.scr1.@closeAbove; sim_idx: 37; agent_idx: 26
        [0, 1],  # comp1.screens.scr1.@lightPollutionPrevention (bool); sim_idx: 38; agent_idx: 27
        [0, 1],  # comp1.screens.scr2.@enabled (bool, unchangeable); sim_idx: 39; agent_idx: 28
        # choose one material 
        [0, 1],  # comp1.screens.scr2.@material == scr_Transparent.par (bool, unchangeable); sim_idx: 40; agent_idx: 29
        [0, 1],  # comp1.screens.scr2.@material == scr_Shade.par (bool, unchangeable); sim_idx: 41; agent_idx: 30
        [0, 1],  # comp1.screens.scr2.@material == scr_Blackout.par (bool, unchangeable); sim_idx: 42; agent_idx: 31
        [-20, 30],  # comp1.screens.scr2.@ToutMax; sim_idx: 43; agent_idx: 32
        # @closeBelow [1] [3] needs to be descending
        [-20, 10],  # comp1.screens.scr2.@closeBelow[0]; sim_idx: 44; agent_idx: 33
        [0, 500],  # comp1.screens.scr2.@closeBelow[1]; sim_idx: 45; agent_idx: 34
        [11, 30],  # comp1.screens.scr2.@closeBelow[2]; sim_idx: 46; agent_idx: 35
        [0, 500],  # comp1.screens.scr2.@closeBelow[3]; sim_idx: 47; agent_idx: 36
        [501, 1500],  # comp1.screens.scr2.@closeAbove; sim_idx: 48; agent_idx: 37
        [0, 1],  # comp1.screens.scr2.@lightPollutionPrevention (bool); sim_idx: 49; agent_idx: 38
        [0, 1],  # comp1.illumination.lmp1.@enabled (bool); sim_idx: 50; agent_idx: 39
        [0, 200],  # comp1.illumination.lmp1.@intensity (unchangeable); sim_idx: 51; agent_idx: 40
        [0, 20],  # comp1.illumination.lmp1.@hoursLight (daily); sim_idx: 52; agent_idx: 41
        # comp1.illumination.lmp1.@endTime = 20; sim_idx: 53
        [100, 400],  # comp1.illumination.lmp1.@maxIglob; sim_idx: 54; agent_idx: 42
        # comp1.illumination.lmp1.@maxPARsum = 50; sim_idx: 55
        [1, 90],  # crp_lettuce.Intkam.management.@plantDensity (daily); sim_idx: 56; agent_idx: 43
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

    def __init__(self, learning=True, model_paths=MODEL_PATHS, ep_path=EP_PATH, op_traces_path = OP_TRACES_PATH):
        self.action_space = gym.spaces.Box(low=self.action_range[:, 0], high=self.action_range[:, 1])
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(self.num_env_params + self.num_op_params,))

        # # loading checkpoint for model_in
        # checkpoint_in = torch.load(model_in_path)
        # # {'op_other_mean': op_other_mean, 'op_other_std':op_other_std, ...}
        # self.norm_data_in = checkpoint_in['norm_data']
        #
        # # model to predict weather inside the greenhouse, per hour
        # self.net_in = Model(norm_data=self.norm_data_in)
        # self.net_in.load_state_dict(checkpoint_in['state_dict'])
        #
        # # loading checkpoint for model_pl
        # checkpoint_pl = torch.load(model_pl_path)
        # # {'op_plant_mean': op_plant_mean, 'op_plant_std':op_plant_std, ...}
        # self.norm_data_pl = checkpoint_pl['norm_data']
        #
        # # model to predict plant properties, per day
        # self.net_pl = ModelPlant()
        # self.net_pl.load_state_dict(checkpoint_pl['state_dict'])

        # loading initial state distribution
        self.traces = np.load(op_traces_path, allow_pickle=True)  # list of trajectory each with shape (T, 20)

        ckpt_paths = get_ensemble_ckpt_paths(model_paths=model_paths, model_id='all')
        self.net = AGCModelEnsemble(self.num_control_params, self.num_env_params,
                                    self.num_op_params, ckpt_paths)
        self.net.eval()
        self.norm_data = self.net.child_models[0].norm_data

        # loading environmental values
        self.env_values = np.load(ep_path).astype(np.float32)  # shape (day*24, 5); e.g. (1680,5) 1680=70*24
        self._max_episode_steps = self.env_values.shape[0] - 1  # the last step need to output ep

        # state features definition
        self.start_iter = 0
        self.iter = 0
        self.trace_idx = 0
        self.op = None
        self.cp_prev = None
        self.agent_cp_daily = None
        self.num_spacings = 0  # TODO：space starts from 1 or 0?
        self.cum_head_m2 = 0
        self.learning = learning

    def parse_action(self, action):
        action[self.bool_indices] = action[self.bool_indices] > 0.5

        # enforce descending order for some of the actions
        for t in self.descending_indices:
            action[t] = -np.sort(-action[t])

        # only pick one screen material
        for t in self.pick_one_indices:
            action[t] = 0
            action[np.random.choice(t, 1)] = 1

        # record prev action
        if self.cp_prev is None:
            self.cp_prev = action
            self.agent_cp_daily = action

        # enforce unchangeable actions
        action[self.unchangeable_indices] = self.cp_prev[self.unchangeable_indices]

        # enforce descending @plantDensity
        action[43] = min(action[43], self.cp_prev[43])

        # record action at the first hour at each day
        if (self.iter - self.start_iter) % 24 == 0:
            self.agent_cp_daily = action
        action[self.daily_indices] = self.agent_cp_daily[self.daily_indices]

        # add fixed actions
        sim_action = self.default_action
        sim_action[self.action_parse_indices] = action

        # sim_action dim=57, action dim=44
        return sim_action[0], sim_action[1:], action

    def step(self, action: np.ndarray):
        # print('action before parse:', action)
        # parse action to model input dim
        # end dim=1, model_action dim=56, agent_action dim=44
        end, model_action, agent_action = self.parse_action(action)

        # if this is the first action query, also copy it as action[0] and update CUM_HEAD_M2
        if self.iter - self.start_iter == 1:
            self.cum_head_m2 += 1 / model_action[-1]

        # update spacing info
        if agent_action[-1] != self.cp_prev[-1]:
            self.num_spacings += 1

        # update CUM_HEAD_M2 at the start of the day
        cum_head_m2_prev = self.cum_head_m2
        if (self.iter - self.start_iter) % 24 == 0:
            self.cum_head_m2 += 1 / model_action[-1]

        # run net
        op_prev = self.traces[self.trace_idx][self.iter] if self.learning else self.op
        self.op = self.net.forward(model_action, self.env_values[self.iter - 1], op_prev)

        # gather state into agent format
        # TODO: is this normalized?
        output_state = self.gather_agent_state(self.env_values[self.iter], self.op)

        # calculate reward and cost with denormalized data
        if (self.iter - self.start_iter) % 24 == 23:
            agent_gain_curr = self.gain(self.op, self.iter - self.start_iter + 1, self.cum_head_m2)
            agent_gain_prev = self.gain(op_prev, self.iter - self.start_iter, cum_head_m2_prev)
            gain_diff = agent_gain_curr - agent_gain_prev
        else:
            gain_diff = 0
            agent_gain_curr = 0
            agent_gain_prev = 0

        fixed_cost, fixed_cost_info = self.fixed_cost(agent_action, self.cp_prev,
                                                      self.iter - self.start_iter + 1, self.num_spacings)
        var_cost, var_cost_info = self.variable_cost(self.env_values[self.iter], self.op)
        cost = fixed_cost + var_cost
        reward = gain_diff - cost

        # save env info for debug
        agent_action = agent_action
        agent_ep_prev = self.env_values[self.iter - 1]
        agent_ep_curr = self.env_values[self.iter]
        agent_op_prev = op_prev
        agent_op_curr = self.op

        agent_reward = reward
        agent_gain_diff = gain_diff
        agent_cost = cost
        agent_cost_fix = fixed_cost
        agent_cost_variable = var_cost

        # update step and prev_action
        self.iter += 1
        self.cp_prev = agent_action

        # end trajectory if (action[0] a.k.a. end is True and fw > 210) or exceed max step
        fw = self.op[OUTPUT_KEYS_TO_INDEX["comp1.Plant.headFW"]]
        done = end and fw > self.min_fw
        done = done or self.iter >= self._max_episode_steps or self.iter >= self.traces[self.trace_idx].shape[0] - 2

        info = {
            'agent_ep_prev': agent_ep_prev,
            'agent_op_prev': agent_op_prev,
            'agent_ob_prev': self.gather_agent_state(self.env_values[self.iter - 1], op_prev),
            'agent_action': agent_action,
            'agent_ep_curr': agent_ep_curr,
            'agent_op_curr': agent_op_curr,
            'agent_reward': agent_reward,
            'agent_gain_diff': agent_gain_diff,
            'agent_gain_curr': agent_gain_curr,
            'agent_gain_prev': agent_gain_prev,
            'agent_cost': agent_cost,
            'agent_cost_fix': agent_cost_fix,
            'agent_cost_variable': agent_cost_variable,

            'agent_cost_occupation': fixed_cost_info[0],
            'agent_cost_fix_co2': fixed_cost_info[1],
            'agent_cost_lamp': fixed_cost_info[2],
            'agent_cost_screen': fixed_cost_info[3],
            'agent_cost_spacing': fixed_cost_info[4],

            'agent_cost_elec': var_cost_info[0],
            'agent_cost_heating': var_cost_info[1],
            'agent_cost_var_co2': var_cost_info[2]
        }

        # TODO: std is 0, normalize to inf
        return output_state, reward, done, info

    def reset(self, start=None):
        # if START is none, randomly choose a start date
        if start is None:
            self.start_iter = np.random.choice(min(self.init_day_range, int(self.traces.shape[0]/24))) * 24\
                             if self.learning else 0
        # otherwise, start from day START
        else:
            self.start_iter = start * 24
        self.iter = self.start_iter

        # randomly choose a trace
        self.trace_idx = np.random.choice(self.traces.shape[0])
        self.op = self.traces[self.trace_idx][self.iter]
        self.cp_prev = None
        self.agent_cp_daily = None

        # spacing info
        self.num_spacings = 0
        self.cum_head_m2 = 0

        output_state = self.gather_agent_state(self.env_values[self.iter], self.op)

        self.iter += 1

        return output_state

    def render(self, mode='human'):
        raise NotImplementedError

    def gather_agent_state(self, ep, op):
        ep = normalize(ep, self.norm_data['ep_mean'], self.norm_data['ep_std'])
        op = normalize(op, self.norm_data['op_mean'], self.norm_data['op_std'])
        return np.concatenate([ep, op])

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
                        val = ';'
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
