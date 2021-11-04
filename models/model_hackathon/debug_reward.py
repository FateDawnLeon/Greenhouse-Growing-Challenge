import json
import os

import matplotlib.pyplot as plt
import numpy as np

from constant import CONTROL_RL, CONTROL_BO
from data import ControlParser
from env import GreenhouseSim
from utils import load_json_data, save_json_data

# matplotlib.use('TkAgg')

data_dir = '../../data/output_trace_full'

ep = np.load(os.path.join(data_dir, 'ep_trace.npy'))
op = np.load(os.path.join(data_dir, 'op_trace.npy'))
pd = np.load(os.path.join(data_dir, 'pd_trace.npy'))
ph = np.load(os.path.join(data_dir, 'ph_trace.npy'))
pl = np.load(os.path.join(data_dir, 'pl_trace.npy'))

control = load_json_data(os.path.join(data_dir, 'control_5.008.json'))
output = load_json_data(os.path.join(data_dir, 'output_5.008.json'))

assert ep.shape[0] % 24 == 0
assert ep.shape[0] == op.shape[0]
assert ep.shape[0] == pd.shape[0]
assert ep.shape[0] == ph.shape[0]
assert ep.shape[0] == pl.shape[0]
num_days = ep.shape[0] // 24

ep = ep.reshape((num_days, 24, -1))
op = op.reshape((num_days, 24, -1))
pl = pl[12::24]
ph = ph.reshape((num_days, 24, -1))
pd = pd[12::24].flatten()

print(ep.shape, op.shape, pl.shape, ph.shape, pd.shape)

control_parser = ControlParser(control)
rl_control = control_parser.parse2dict([k for k in CONTROL_RL.keys() if k != 'end'])
# bo_control = control_parser.parse2dict(CONTROL_BO[1:-1])
# bo_control = {k: v[0, 0, 0] for k, v in bo_control.items()}
# bo_control['init_plant_density'] = pd[0]
# save_json_data(bo_control, 'runtime_data/BO.json')
rl_control = {k: v.squeeze() for k, v in rl_control.items()}
pd_val = np.array([90 - pd[0]] + [pd[i - 1] - pd[i] for i in range(1, num_days)])[:, np.newaxis]
pd_change = np.array([90 != pd[0]] + [pd[i] != pd[i - 1] for i in range(1, num_days)])[:, np.newaxis]
rl_control['crp_lettuce.Intkam.management.@plantDensity'] = np.hstack((pd_val, pd_change))
print(rl_control['crp_lettuce.Intkam.management.@plantDensity'])

# start testing
gains = np.zeros(num_days)

cost_plant = np.zeros(num_days)
cost_occupation = np.zeros(num_days)
cost_fix_co2 = np.zeros(num_days)
cost_lamp = np.zeros(num_days)
cost_screen = np.zeros(num_days)
cost_spacing = np.zeros(num_days)

cost_elec = np.zeros(num_days)
cost_heating = np.zeros(num_days)
cost_var_co2 = np.zeros(num_days)

cum_head_m2 = 0
num_spacings = 0
for i in range(num_days):
    action_dict = {k: v[i] for k, v in rl_control.items()}

    cum_head_m2 += 1 / pd[i]
    num_spacings += action_dict['crp_lettuce.Intkam.management.@plantDensity'][1]
    if i == 0:
        delta_avg_head_m2 = (i + 1) / cum_head_m2 - (i + 0)
    else:
        delta_avg_head_m2 = (i + 1) / cum_head_m2 - (i + 0) / (cum_head_m2 - 1 / pd[i])

    gains[i] = GreenhouseSim.gain(pl[i], i, cum_head_m2, training=False)

    cost_plant[i] = GreenhouseSim.fixed_cost(action_dict, i, num_spacings, delta_avg_head_m2)[1][0]
    cost_occupation[i] = GreenhouseSim.fixed_cost(action_dict, i, num_spacings, delta_avg_head_m2)[1][1]
    cost_fix_co2[i] = GreenhouseSim.fixed_cost(action_dict, i, num_spacings, delta_avg_head_m2)[1][2]
    cost_lamp[i] = GreenhouseSim.fixed_cost(action_dict, i, num_spacings, delta_avg_head_m2)[1][3]
    cost_screen[i] = GreenhouseSim.fixed_cost(action_dict, i, num_spacings, delta_avg_head_m2)[1][4]
    cost_spacing[i] = GreenhouseSim.fixed_cost(action_dict, i, num_spacings, delta_avg_head_m2)[1][5]

    cost_elec[i] = GreenhouseSim.variable_cost(ph[i], op[i])[1][0]
    cost_heating[i] = GreenhouseSim.variable_cost(ph[i], op[i])[1][1]
    cost_var_co2[i] = GreenhouseSim.variable_cost(ph[i], op[i])[1][2]

print(f'avg_head_m2={num_days / cum_head_m2:.3f}', output['stats']['economics']['info']['AverageHeadm2'])
print(f'{gains[-1]=:.3f}', output['stats']['economics']['gains']['total'])
print()
print(f'{sum(cost_plant)=:.3f}', output['stats']['economics']['fixedCosts']['objects']['plants'])
print(f'{sum(cost_occupation)=:.3f}', output['stats']['economics']['fixedCosts']['objects']['comp1.Greenhouse.Costs'])
print(f'{sum(cost_fix_co2)=:.3f}', output['stats']['economics']['fixedCosts']['objects']['comp1.ConCO2.Costs'])
print(f'{sum(cost_lamp)=:.3f}', output['stats']['economics']['fixedCosts']['objects']['comp1.Lmp1.Costs'])
true_screen_cost = 0
if 'comp1.Scr1.Costs' in output['stats']['economics']['fixedCosts']['objects'].keys():
    true_screen_cost += output['stats']['economics']['fixedCosts']['objects']['comp1.Scr1.Costs']
if 'comp1.Scr2.Costs' in output['stats']['economics']['fixedCosts']['objects'].keys():
    true_screen_cost += output['stats']['economics']['fixedCosts']['objects']['comp1.Scr2.Costs']
print(f'{sum(cost_screen)=:.3f}', true_screen_cost)
print(f'{sum(cost_spacing)=:.3f}', output['stats']['economics']['fixedCosts']['objects']['spacingSystem'])
print()
print(f'{sum(cost_elec)=:.3f}', output['stats']['economics']['variableCosts']['objects']['comp1.Lmp1.ElecUse'])
print(f'{sum(cost_heating)=:.3f}', output['stats']['economics']['variableCosts']['objects']['comp1.Pipe1.GasUse'])
print(f'{sum(cost_var_co2)=:.3f}', output['stats']['economics']['variableCosts']['objects']['CO2'])
