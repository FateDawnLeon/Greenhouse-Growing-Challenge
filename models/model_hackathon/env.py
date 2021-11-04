from collections import OrderedDict
from datetime import datetime
import os

import gym
import numpy as np
import torch

from constant import CLIMATE_MODEL_PATH, PLANT_MODEL_PATH, FULL_EP_PATH, FULL_PEAKHOUR_PATH, TRACES_DIR, \
    BO_CONTROL_PATH, \
    EP_KEYS, OP_KEYS, PL_KEYS, PL_INIT_VALUE, ACTION_PARAM_SPACE, BOOL_ACTION_IDX, PD_DELTA_IDX, \
    INDEX_OP_TO_OP_IN, EARLIEST_START_DATE
from constant import get_range
from model import get_model
from data import parse_action, agent_action_to_dict, agent_action_to_array
from utils import list_keys_to_index, load_json_data

EP_INDEX = list_keys_to_index(EP_KEYS)
OP_INDEX = list_keys_to_index(OP_KEYS)
PL_INDEX = list_keys_to_index(PL_KEYS)

BO_CONTROLS = load_json_data(BO_CONTROL_PATH)


class GreenhouseSim(gym.Env):
    MIN_PD = 5
    MIN_FW = 210

    num_ep = len(EP_KEYS) * 24
    num_op = len(OP_KEYS) * 24
    num_pl = len(PL_KEYS)

    def __init__(self, training=True, climate_model_paths=CLIMATE_MODEL_PATH, plant_model_path=PLANT_MODEL_PATH,
                 full_ep_path=FULL_EP_PATH, full_peakhour_path=FULL_PEAKHOUR_PATH, traces_dir=TRACES_DIR):
        action_lo, action_hi = get_range(ACTION_PARAM_SPACE.keys(), ACTION_PARAM_SPACE)
        self.action_space = gym.spaces.Box(low=action_lo, high=action_hi)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(self.num_ep + self.num_op + self.num_pl + 1,))

        self.training = training

        if self.training:
            # load initial state distribution
            # traces dir: TRACES_DIR/{IDX}/ep_trace.npy
            #             TRACES_DIR/{IDX}/peakhour_trace.npy
            #             TRACES_DIR/{IDX}/op_trace.npy
            #             TRACES_DIR/{IDX}/pl_trace.npy
            #             TRACES_DIR/{IDX}/pd_trace.npy
            self.trace_paths = [f'{traces_dir}/{f}' for f in os.listdir(traces_dir) if
                                os.path.isdir(f'{traces_dir}/{f}')]
            self.op_trace = None
            self.pl_trace = None
            self.pd_trace = None
        else:
            # load full ep&peakhour trace
            self.full_peakhour_trace = np.load(full_peakhour_path)

        self.full_ep_trace = np.load(full_ep_path)
        self.ep_trace = None
        self.peakhour_trace = None

        self._max_episode_steps = self.full_ep_trace.shape[0] - 1

        # load model
        self.climate_model = get_model(climate_model_paths)
        self.plant_model = get_model(plant_model_path)

        # state features definition
        self.ep = None
        self.op = None
        self.pl = None
        self.pd = None

        self.iter = None
        self.cum_head_m2 = None
        self.num_spacings = None

    def reset(self):
        if self.training:
            # randomly choose a trace
            trace_idx = np.random.choice(len(self.trace_paths))
            # EP trace: shape (num_days, 24, NUM_EP_PARAMS)
            self.ep_trace = np.load(os.path.join(self.trace_paths[trace_idx], 'ep_trace.npy'))
            # PEAKHOUR trace: shape (num_days, 24, 1)
            self.peakhour_trace = np.load(os.path.join(self.trace_paths[trace_idx], 'ph_trace.npy'))
            # OP trace: shape (num_days, 24, NUM_OP_PARAMS)
            self.op_trace = np.load(os.path.join(self.trace_paths[trace_idx], 'op_trace.npy'))
            # PL trace: shape (num_days, NUM_PL_PARAMS)
            self.pl_trace = np.load(os.path.join(self.trace_paths[trace_idx], 'pl_trace.npy'))
            # PD trace: shape (num_days, 1)
            self.pd_trace = np.load(os.path.join(self.trace_paths[trace_idx], 'pd_trace.npy'))
        else:
            start_day = datetime.strptime(BO_CONTROLS['simset.@startDate'], '%Y-%m-%d').date()
            start_day = (start_day - EARLIEST_START_DATE).days
            self.ep_trace = self.full_ep_trace[start_day:]
            self.peakhour_trace = self.full_peakhour_trace[start_day:]

        self.ep = self.ep_trace[0]
        self.op = np.zeros((24, self.num_op // 24))
        self.pl = agent_action_to_array(PL_INIT_VALUE)
        self.pd = BO_CONTROLS['init_plant_density']

        self.iter = 0
        self.cum_head_m2 = 1. / BO_CONTROLS['init_plant_density']
        self.num_spacings = 0  # number of spacing __CHANGES__

        state = np.concatenate((self.ep, self.op, self.pl, self.pd), axis=None, dtype=np.float32)  # flatten
        return state

    @staticmethod
    def bool_action(action: np.ndarray) -> np.ndarray:
        # force bool on some values
        action[BOOL_ACTION_IDX] = np.round(action[BOOL_ACTION_IDX])

        return action

    def step(self, action: np.ndarray):
        action = self.bool_action(action)
        action_dict = agent_action_to_dict(action)

        # calculate new values for some features
        density_tuple = action_dict['crp_lettuce.Intkam.management.@plantDensity']
        density_delta = density_tuple[0] if density_tuple[1] else 0.0
        plant_density_new = self.pd - density_delta
        plant_density_new = np.maximum(plant_density_new, self.MIN_PD)
        action[PD_DELTA_IDX] = self.pd - plant_density_new
        if plant_density_new == self.pd:
            action_dict['crp_lettuce.Intkam.management.@plantDensity'][1] = 0
        cum_head_m2_new = self.cum_head_m2 + 1. / plant_density_new
        num_spacings_new = self.num_spacings + density_tuple[1]
        if self.iter == 0:
            delta_avg_head_m2 = (self.iter + 1) / cum_head_m2_new
        else:
            delta_avg_head_m2 = (self.iter + 1) / cum_head_m2_new - self.iter / self.cum_head_m2

        # update @plantDensity from relative to absolute value
        model_action_dict = action_dict.copy()
        model_action_dict['crp_lettuce.Intkam.management.@plantDensity'] = np.array([plant_density_new])
        model_action_dict.update(BO_CONTROLS)
        model_action_dict['day_offset'] = self.iter
        model_action_dict['comp1.illumination.lmp1.@enabled'] = BO_CONTROLS['comp1.illumination.lmp1.@intensity'] > 0

        # use model to predict next state
        # op_{d+1} = ModelClimate(cp_{d+1}, ep_{d+1}, op_{d})
        op_new = self.climate_model.predict(parse_action(model_action_dict), self.ep, self.op)
        # op^in_{d+1} = select(op_{d+1})
        op_in_new = op_new[:, INDEX_OP_TO_OP_IN]
        # pl_{d+1} = ModelPlant(pd_{d+1}, op^in_{d+1}, pl_{d})
        pl_new = self.plant_model.predict(np.array([plant_density_new]), op_in_new, self.pl)

        output_state = np.concatenate((self.ep_trace[self.iter + 1], op_new, pl_new, np.array([plant_density_new])),
                                      axis=None, dtype=np.float32)

        # compute reward
        gain_curr = self.gain(pl_new, self.iter + 1, cum_head_m2_new, self.training)
        gain_prev = self.gain(self.pl, self.iter, self.cum_head_m2, self.training)
        fixed_cost, _ = self.fixed_cost(action_dict, self.iter + 1, num_spacings_new, delta_avg_head_m2)
        variable_cost, _ = self.variable_cost(self.peakhour_trace[self.iter], op_new)
        reward = gain_curr - gain_prev - fixed_cost - variable_cost

        # update those features
        self.cum_head_m2 = cum_head_m2_new
        self.num_spacings = cum_head_m2_new

        done = action[0].squeeze() and pl_new[PL_INDEX['comp1.Plant.headFW']] > self.MIN_FW
        done = done or self.iter == len(self.ep_trace) - 2

        info = {'parsed_action': action}

        self.ep = self.ep_trace[self.iter + 1]
        if self.training:
            info = {
                # only meaningful if SELF.TRAINING = True
                'current_real_observation': np.concatenate((self.ep, self.op, self.pl, self.pd), axis=None,
                                                           dtype=np.float32)
            }
            # update state to real next state in the trace
            self.op = self.op_trace[self.iter]
            self.pl = self.pl_trace[self.iter]
            self.pd = self.pd_trace[self.iter][0]
        else:
            # update state to predicted next state
            self.op = op_new
            self.pl = pl_new
            self.pd = plant_density_new

        self.iter += 1

        return output_state, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError

    @staticmethod
    def gain(pl, it, cum_head_m2, training):
        fw = pl[PL_INDEX['comp1.Plant.headFW']]
        dmc = pl[PL_INDEX['comp1.Plant.shootDryMatterContent']]
        quality_loss = pl[PL_INDEX['comp1.Plant.qualityLoss']]

        if training:  # use a slightly different profit curve during training
            if fw <= 230:
                price = 0.4 * fw / 230
            elif fw <= 250:
                price = 0.4 + 0.1 * (fw - 230) / (250 - 230)
            elif fw <= 270:
                price = 0.5 - 0.1 * (fw - 250) / (270 - 250)
            elif fw <= 290:
                price = 0.4 - 0.4 * (fw - 270) / (290 - 270)
            else:  # fw > 290
                price = 0
        else:
            # mirror fresh weight
            fw = fw if fw <= 250 else 500 - fw
            if fw <= 210:
                price = 0
            elif fw <= 230:
                price = 0.4 * (fw - 210) / (230 - 210)
            elif fw <= 250:
                price = 0.4 + 0.1 * (fw - 230) / (250 - 230)
            else:
                print(f'{fw=}, how is it fking possible that this is more than 250???')
                raise ValueError

        # adjust for dmc
        if dmc < 0.045:
            price *= 0.9
        elif dmc > 0.05:
            price *= 1.1

        # adjust for quality loss
        price *= (1 - 0.01 * quality_loss)

        # adjust for density
        avg_head_m2 = (it + 1) / cum_head_m2
        price *= avg_head_m2
        return price

    @staticmethod
    def fixed_cost(action_dict, it, num_spacings, delta_avg_head_m2):
        # plant cost
        cost_plant = delta_avg_head_m2 * 0.12
        # greenhouse occupation
        cost_occupation = 17 / 365
        # CO2 dosing capacity
        cost_fix_co2 = BO_CONTROLS['common.CO2dosing.@pureCO2cap'] * 0.015 / 365
        # lamp maintenance
        cost_lamp = BO_CONTROLS['comp1.illumination.lmp1.@intensity'] * 0.0393 / 365
        # screen usage
        cost_screen = (int(BO_CONTROLS['comp1.screens.scr1.@enabled'])
                       + int(BO_CONTROLS['comp1.screens.scr2.@enabled'])) * 1.25 / 365
        # spacing changes
        if action_dict['crp_lettuce.Intkam.management.@plantDensity'][1]:
            cost_spacing = it * 1.5 / 365
        else:
            cost_spacing = 0
        cost_spacing += num_spacings * 1.5 / 365

        cost_total = cost_plant + cost_occupation + cost_fix_co2 + cost_lamp + cost_screen + cost_spacing
        return cost_total, (cost_plant, cost_occupation, cost_fix_co2, cost_lamp, cost_screen, cost_spacing)

    @staticmethod
    def variable_cost(peakhour, op):
        # electricity cost
        electricity = op[:, OP_INDEX['comp1.Lmp1.ElecUse']]
        cost_elec = electricity / 1000 * np.where(peakhour.squeeze() > 0.5, 0.1, 0.06)
        # heating cost
        pipe_value = op[:, OP_INDEX['comp1.PConPipe1.Value']]
        cost_heating = pipe_value / 1000 * 0.03
        # CO2 cost
        pure_air_value = op[:, OP_INDEX['comp1.McPureAir.Value']]
        cost_var_co2 = pure_air_value * 3600 * 0.12

        cost_total = np.sum(cost_elec + cost_heating + cost_var_co2)

        return cost_total, (np.sum(cost_elec), np.sum(cost_heating), np.sum(cost_var_co2))
