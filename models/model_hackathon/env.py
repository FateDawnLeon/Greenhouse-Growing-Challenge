import os
from collections import OrderedDict

import gym
import numpy as np
import torch

from constant import CONTROL_INFO, CLIMATE_MODEL_PATH, PLANT_MODEL_PATH, CLIMATE_NORM_DATA, PLANT_NORM_DATA, \
    EP_PATH, TRACES_DIR, INIT_PLANT_DENSITY, BO_CONTROLS
from data import ClimateDatasetDay, PlantDatasetDay
from model import ClimateModelDay, PlantModelDay
from utils import list_keys_to_index

EP_INDEX = list_keys_to_index(ClimateDatasetDay.EP_KEYS)
OP_INDEX = list_keys_to_index(ClimateDatasetDay.OP_KEYS)
PL_INDEX = list_keys_to_index(PlantDatasetDay.OP_PL_KEYS)


class GreenhouseSim(gym.Env):
    num_cp = 16
    num_ep = len(ClimateDatasetDay.EP_KEYS) * 24
    num_op = len(ClimateDatasetDay.OP_KEYS) * 24
    num_op_in = len(PlantDatasetDay.OP_IN_KEYS) * 24
    num_pl = len(PlantDatasetDay.OP_PL_KEYS)

    action_range = np.array([
        [0, 1],  # end
        [10, 15],  # comp1.setpoints.temp.@heatingTemp - night
        [15, 30],  # comp1.setpoints.temp.@heatingTemp - day
        [0, 5],  # comp1.setpoints.temp.@ventOffset
        [0, 50],  # comp1.setpoints.ventilation.@startWnd
        [200, 800],  # comp1.setpoints.CO2.@setpoint - night
        [800, 1200],  # comp1.setpoints.CO2.@setpoint - day
        [-20, 30],  # comp1.screens.scr1.@ToutMax
        [0, 200],  # comp1.screens.scr1.@closeBelow
        [1000, 1500],  # comp1.screens.scr1.@closeAbove
        [-20, 30],  # comp1.screens.scr2.@ToutMax
        [0, 200],  # comp1.screens.scr2.@closeBelow
        [1000, 1500],  # comp1.screens.scr2.@closeAbove
        [16, 20],  # comp1.illumination.lmp1.@hoursLight
        [0, 50],  # crp_lettuce.Intkam.management.@plantDensity
    ], dtype=np.float32)

    def __init__(self, training=False, climate_model_paths=CLIMATE_MODEL_PATH, plant_model_path=PLANT_MODEL_PATH,
                 full_ep_path=EP_PATH, traces_dir=TRACES_DIR):
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
        self.climate_model.load_state_dict(torch.load(climate_model_paths))
        self.climate_model.eval()
        self.plant_model = PlantModelDay(self.num_op_in, self.num_pl, PLANT_NORM_DATA)
        self.plant_model.load_state_dict(torch.load(plant_model_path))
        self.plant_model.eval()

        # state features definition
        self.ep = None
        self.op = None
        self.pl = None

        self.iter = 0
        self.plant_density = INIT_PLANT_DENSITY
        self.cum_head_m2 = 0
        self.num_spacings = 0

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
        self.plant_density = INIT_PLANT_DENSITY
        self.cum_head_m2 = 0
        self.num_spacings = 0

        state = np.concatenate((self.ep, self.op, self.pl), axis=None)  # flatten
        return state

    def step(self, action: np.ndarray):
        action_dict = self.agent_action_to_dict(action)

        # use model to predict next state
        # op_{d+1} = ModelClimate(cp_{d+1}, ep_{d+1}, op_{d})
        op_new = self.climate_model.predict(action, self.ep, self.op)
        # op^in_{d+1} = select(op_{d+1})
        op_in_new = op_new[:, PlantDatasetDay.INDEX_OP_TO_OP_IN]
        # pl_{d+1} = ModelPlant(op^in_{d+1}, pl_{d})
        pl_new = self.plant_model(op_in_new, self.pl)

        # calculate new values for some features
        plant_density_new = self.plant_density - action_dict['crp_lettuce.Intkam.management.@plantDensity']
        cum_head_m2_new = self.cum_head_m2 + 1. / plant_density_new
        num_spacings_new = self.num_spacings + int(action_dict['crp_lettuce.Intkam.management.@plantDensity'] != 0)

        output_state = np.concatenate((self.ep_trace[self.iter + 1], op_new, pl_new))

        # compute reward
        gain_curr = self.gain(pl_new, self.iter + 1, cum_head_m2_new)
        gain_prev = self.gain(self.pl, self.iter, self.cum_head_m2)
        fixed_cost, _ = self.fixed_cost(action_dict, self.iter + 1, num_spacings_new)
        variable_cost, _ = self.variable_cost(self.ep_trace[self.iter], op_new)
        reward = gain_curr - gain_prev - fixed_cost - variable_cost

        # update those features
        self.plant_density = plant_density_new
        self.cum_head_m2 = cum_head_m2_new
        self.num_spacings = cum_head_m2_new

        done = action[0] or self.iter == len(self.ep_trace) - 1

        self.ep = self.ep_trace[self.iter + 1]
        if self.training:
            # update state to real next state in the trace
            self.op = self.op_trace[self.iter]
            self.pl = self.pl_trace[self.iter]
        else:
            # update state to predicted next state
            self.op = op_new
            self.pl = pl_new

        self.iter += 1

        info = {
            # only meaningful if SELF.TRAINING = True
            'real_next_state': np.concatenate((self.ep, self.op, self.pl), axis=None)
        }

        return output_state, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError

    @staticmethod
    def gain(pl, it, cum_head_m2):
        fw = pl[PL_INDEX['comp1.Plant.headFW']]
        dmc = pl[PL_INDEX['comp1.Plant.shootDryMatterContent']]
        quality_loss = pl[PL_INDEX['comp1.Plant.qualityLoss']]

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

        # adjust for quality loss
        price *= (1 - 0.01 * quality_loss)

        # adjust for density
        num_days = it + 1
        avg_head_m2 = num_days / cum_head_m2
        price *= avg_head_m2

        return price

    @staticmethod
    def fixed_cost(action_dict, it, num_spacings):
        # greenhouse occupation
        cost_occupation = 11.5 / 365
        # CO2 dosing capacity
        cost_fix_co2 = BO_CONTROLS['common.CO2dosing.@pureCO2cap'] * 0.015 / 365
        # lamp maintenance
        cost_lamp = BO_CONTROLS['comp1.illumination.lmp1.@intensity'] * 0.0281 / 365
        # screen usage
        cost_screen = (int(BO_CONTROLS['comp1.screens.scr1.@enabled'])
                       + int(BO_CONTROLS['comp1.screens.scr2.@enabled'])) * 0.75 / 365
        # spacing changes
        if action_dict['crp_lettuce.Intkam.management.@plantDensity'] != 0:
            cost_spacing = (it - 1) * 1.5 / 365
        else:
            cost_spacing = 0
        cost_spacing += num_spacings * 1.5 / 365

        cost_total = cost_occupation + cost_fix_co2 + cost_lamp + cost_screen + cost_spacing
        return cost_total, (cost_occupation, cost_fix_co2, cost_lamp, cost_screen, cost_spacing)

    @staticmethod
    def variable_cost(ep, op):
        # electricity cost
        peak_hour = ep[EP_INDEX['common.Economics.PeakHour']]
        electricity = op[OP_INDEX['comp1.Lmp1.ElecUse']]
        if peak_hour > 0.5:
            cost_elec = electricity / 1000 * 0.1
        else:
            cost_elec = electricity / 1000 * 0.06
        # heating cost
        pipe_value = op[OP_INDEX['comp1.PConPipe1.Value']]
        cost_heating = pipe_value / 1000 * 0.03
        # CO2 cost
        pure_air_value = op[OP_INDEX['comp1.McPureAir.Value']]
        cost_var_co2 = pure_air_value * 3600 * 0.12

        cost_total = cost_elec + cost_heating + cost_var_co2

        return cost_total, (cost_elec, cost_heating, cost_var_co2)
