import os
import json
from random import random
import requests
import datetime
import numpy as np

from astral.sun import sun
from torch.utils.data import Dataset

from control_param import ControlParamSimple
from constant import CITY, COMMON_DATA_DIR, CONTROL_KEYS, ENV_KEYS, KEYS, NORM_DATA_PATHS, OUTPUT_KEYS, START_DATE, URL, EP_PATHS, INIT_STATE_PATHS


def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def preprocess_str(param):
    if '.par' in param:
        return param == np.array(
            ['scr_Transparent.par', 'scr_Shade.par', 'scr_Blackout.par']
        )

    elif ';' in param: # this is a table
        numbers = [[float(x) for x in y.split()] for y in param.split(';')]
        return np.asarray(numbers).flatten()
    else: # this is a number list
        numbers = [float(x) for x in param.split()]
        return np.asarray(numbers)


def preprocess_screen_threshold(param, num_hours):
    if isinstance(param, str):
        val = preprocess_str(param)
    else:
        val = np.asarray([0, param, 10, param])
    return np.repeat(val[:, np.newaxis], num_hours, axis=1)


def preprocess_plant_density(param, num_hours):
    spacing_scheme = [[int(x) for x in y.split()] for y in param.split(';')]
    densities = np.zeros(num_hours, dtype=np.int)
    for day, density in spacing_scheme:
        t = (day - 1) * 24
        densities[t:] = density
    return densities[np.newaxis,:]


def preprocess_pipe_radinf(param, num_hours):
    numbers = [float(x) for x in param.split()]
    if len(numbers) == 1:
        assert numbers[0] == 0
        numbers *= 2
    val = np.asarray(numbers)
    return np.repeat(val[:, np.newaxis], num_hours, axis=1)


def preprocess_co2_dose_capacity(param, num_hours):
    val = preprocess_str(param)
    if len(val) == 1:
        val = np.array([20, val[0], 40, val[0], 70, val[0]])
    return np.repeat(val[:, np.newaxis], num_hours, axis=1)


special_preprocess = {
    'comp1.screens.scr1.@closeBelow': preprocess_screen_threshold,
    'comp1.screens.scr2.@closeBelow': preprocess_screen_threshold,
    'crp_lettuce.Intkam.management.@plantDensity': preprocess_plant_density,
    'comp1.heatingpipes.pipe1.@radiationInfluence': preprocess_pipe_radinf,
    'comp1.setpoints.CO2.@doseCapacity': preprocess_co2_dose_capacity,
}


def parse_static_param(key, param, num_hours):
    if key in special_preprocess:
        return special_preprocess[key](param, num_hours)
    if type(param) in [int, float, bool]:
        val = np.array([param])
    elif type(param) == str:
        val = preprocess_str(param)
    else:
        raise ValueError(f'praram {key}:{param} data type not supported!')
    return np.repeat(val[:, np.newaxis], num_hours, axis=1)


def parse_dynamic_param(key, param, num_hours):  # sourcery no-metrics
    value_scheme = []
    for date in param:
        day, month = [int(x) for x in date.split('-')]
        dateinfo = datetime.date(2021, month, day)

        day_param = param[date]
        if isinstance(day_param, dict): # this is a 24-hour schedule
            s = sun(CITY.observer, date=dateinfo, tzinfo=CITY.timezone)
            t_sunrise = s['sunrise'].hour + s['sunrise'].minute / 60
            t_sunset = s['sunset'].hour + s['sunset'].minute / 60
            time_vals = []
            for time in day_param:
                val = day_param[time]
                time = time.replace('r', str(round(t_sunrise, 1)))
                time = time.replace('s', str(round(t_sunset, 1)))
                time_vals.append((round(eval(time)), val))

            if len(time_vals) == 1:
                day_values = [time_vals[0][1]] * 24
            else:
                day_values = [None] * 24
                for cur in range(len(time_vals)):
                    left_time, left_val = time_vals[cur]
                    right_time, right_val = time_vals[(cur+1) % len(time_vals)]

                    if right_time < left_time:
                        right_time += 24

                    d_v = right_val - left_val
                    d_t = right_time - left_time
                    for i in range(left_time, right_time):
                        day_values[i%24] = left_val + (i - left_time) * d_v / d_t
            day_values = np.array(day_values)
            day_values = day_values[np.newaxis, :]
        else:
            day_values = parse_static_param(key, day_param, 24)

        delta = dateinfo - START_DATE
        day_offset = max(0, delta.days)
        value_scheme.append((day_offset, day_values))

    n = value_scheme[0][1].shape[0]
    values = np.zeros((n, num_hours))
    for day_offset, day_values in value_scheme:
        t = day_offset * 24
        day_left = (num_hours - t) // 24
        values[:, t:] = np.concatenate([day_values]*day_left, axis=1)

    return values


def parse_control(control):
    end_date = datetime.date.fromisoformat(control['simset']['@endDate'])
    num_days = (end_date - START_DATE).days
    num_hours = num_days * 24

    control_vals = []
    for key in CONTROL_KEYS[1:]:
        key_path = key.split('.')
        param = control
        for path in key_path:
            param = param[path]

        if isinstance(param, dict):
            values = parse_dynamic_param(key, param, num_hours)
        else:
            values = parse_static_param(key, param, num_hours)

        # print(key, values.shape)
        assert values.shape[1] == num_hours
        control_vals.append(values)
    
    control_vals = np.concatenate(control_vals, axis=0) # M x T
    control_vals = control_vals.T # T x M

    return control_vals


def parse_output(output):
    env_vals = []
    for key in ENV_KEYS:
        val = output['data'][key]['data']
        env_vals.append(val)
    env_vals = np.array(env_vals)
    env_vals = env_vals.T # T x N1
    
    output_vals = []
    for key in OUTPUT_KEYS:
        val = output['data'][key]['data']
        val = [-1.0 if x == 'NaN' else x for x in val]
        output_vals.append(val)
    output_vals = np.array(output_vals)
    output_vals = output_vals.T # T x N2

    return env_vals, output_vals


def preprocess_data(data_dir, save=True):
    control_dir = os.path.join(data_dir, 'controls')
    output_dir = os.path.join(data_dir, 'outputs')

    assert os.path.exists(control_dir)
    assert os.path.exists(output_dir)

    cp_names = os.listdir(control_dir)
    op_names = os.listdir(output_dir)
    names = list(set(cp_names).intersection(set(op_names)))

    cp_all, ep_all, op_all = [], [], []
    cp_array, ep_array, op_pre_array, op_cur_array = [], [], [], []
    for name in names:
        control = load_json_data(os.path.join(control_dir, name))
        output = load_json_data(os.path.join(output_dir, name))

        assert output['responsemsg'] == 'ok'

        control_vals = parse_control(control) # control_vals: T x M
        env_vals, output_vals = parse_output(output) # env_vals: T x N1, output_vals: T x N2

        cp_all.append(control_vals)
        ep_all.append(env_vals)
        op_all.append(output_vals)
        
        cp_vals = control_vals[:-1]
        ep_vals = env_vals[:-1]
        op_vals_pre = output_vals[:-1]
        op_vals_cur = output_vals[1:]

        assert cp_vals.shape[0] == ep_vals.shape[0] == op_vals_pre.shape[0] == op_vals_cur.shape[0]

        cp_array.append(cp_vals)
        ep_array.append(ep_vals)
        op_pre_array.append(op_vals_pre)
        op_cur_array.append(op_vals_cur)
    
    cp_array = np.concatenate(cp_array, axis=0)
    ep_array = np.concatenate(ep_array, axis=0)
    op_pre_array = np.concatenate(op_pre_array, axis=0)
    op_cur_array = np.concatenate(op_cur_array, axis=0)

    if save:
        np.savez_compressed(f'{data_dir}/processed_data.npz', 
                cp=cp_array, ep=ep_array, op_pre=op_pre_array, op_cur=op_cur_array)

    return cp_all, ep_all, op_all


def zscore_normalize(data_arr, mean_arr, std_arr):
    norm_arr = (data_arr - mean_arr) / std_arr
    return np.nan_to_num(norm_arr)


def zscore_denormalize(norm_arr, mean_arr, std_arr):
    return norm_arr * std_arr + mean_arr


def compute_mean_std(data_dirs):
    cp_all, ep_all, op_all = [], [], []
    
    for data_dir in data_dirs:
        cp, ep, op = preprocess_data(data_dir, save=False)
        cp_all.extend(cp)
        ep_all.extend(ep)
        op_all.extend(op)
    
    cp_all = np.concatenate(cp_all, axis=0)
    ep_all = np.concatenate(ep_all, axis=0)
    op_all = np.concatenate(op_all, axis=0)

    cp_mean, cp_std = np.mean(cp_all, axis=0, dtype=np.float32), np.std(cp_all, axis=0, dtype=np.float32)
    ep_mean, ep_std = np.mean(ep_all, axis=0, dtype=np.float32), np.std(ep_all, axis=0, dtype=np.float32)
    op_mean, op_std = np.mean(op_all, axis=0, dtype=np.float32), np.std(op_all, axis=0, dtype=np.float32)

    return (cp_mean, cp_std), (ep_mean, ep_std), (op_mean, op_std)


class SupervisedModelDataset(Dataset):
    def __init__(self, data_dirs, normalize=False):
        super(SupervisedModelDataset, self).__init__()

        if normalize:
            (self.cp_mean, self.cp_std), (self.ep_mean, self.ep_std), (self.op_mean, self.op_std) = compute_mean_std(data_dirs)

        self.normalize = normalize

        cp_arr_list, ep_arr_list = [], []
        op_pre_arr_list, op_cur_arr_list = [], []
        
        for data_dir in data_dirs:
            preprocess_data(data_dir)
            data_path = f'{data_dir}/processed_data.npz'
            data = np.load(data_path)

            cp_arr_list.append(data['cp'])
            ep_arr_list.append(data['ep'])
            op_pre_arr_list.append(data['op_pre'])
            op_cur_arr_list.append(data['op_cur'])

        self.cp = np.concatenate(cp_arr_list, axis=0, dtype=np.float32)
        self.ep = np.concatenate(ep_arr_list, axis=0, dtype=np.float32)
        self.op_pre = np.concatenate(op_pre_arr_list, axis=0, dtype=np.float32)
        self.op_cur = np.concatenate(op_cur_arr_list, axis=0, dtype=np.float32)

        assert self.cp.shape[0] == self.ep.shape[0] == self.op_pre.shape[0]
        assert self.op_pre.shape == self.op_cur.shape

        if self.normalize:
            self.cp = zscore_normalize(self.cp, self.cp_mean, self.cp_std)
            self.ep = zscore_normalize(self.ep, self.ep_mean, self.ep_std)
            self.op_pre = zscore_normalize(self.op_pre, self.op_mean, self.op_std)
            self.op_cur = zscore_normalize(self.op_cur, self.op_mean, self.op_std)

    def __getitem__(self, index):
        cp_vec = self.cp[index]
        ep_vec = self.ep[index]
        op_pre_vec = self.op_pre[index]
        op_cur_vec = self.op_cur[index]
        return cp_vec, ep_vec, op_pre_vec, op_cur_vec

    def __len__(self):
        return self.cp.shape[0]

    @property
    def cp_dim(self):
        return self.cp.shape[1]
    
    @property
    def ep_dim(self):
        return self.ep.shape[1]
    
    @property
    def op_dim(self):
        return self.op_pre.shape[1]


def get_output(control, sim_id):
    data = {"key": KEYS[sim_id], "parameters": json.dumps(control)}
    headers = {'ContentType': 'application/json'}

    while True:
        response = requests.post(URL, data=data, headers=headers, timeout=300)
        output = response.json()
        print(response, output['responsemsg'])

        if output['responsemsg'] == 'ok':
            break
        elif output['responsemsg'] == 'busy':
            continue
        else:
            raise ValueError('response message not expected!')
    
    return output


def get_op1_pool(data_dirs):
    op_1_pool = []
    for data_dir in data_dirs:
        control_dir = os.path.join(data_dir, 'controls')
        output_dir = os.path.join(data_dir, 'outputs')

        cp_names = os.listdir(control_dir)
        op_names = os.listdir(output_dir)
        names = list(set(cp_names).intersection(set(op_names)))

        for name in names:
            output = load_json_data(os.path.join(output_dir, name))
            assert output['responsemsg'] == 'ok'
            op_1 = [output['data'][key]['data'][0] for key in OUTPUT_KEYS]
            op_1_pool.append(op_1)

    return np.array(op_1_pool, dtype=np.float32)


def get_ep_ndays(sim_id, num_days=60):
    CP = ControlParamSimple()
    CP.set_endDate(num_days=num_days)

    output = get_output(CP.data, sim_id)

    env_vals = []
    for key in ENV_KEYS:
        val = output['data'][key]['data']
        env_vals.append(val)
    env_vals = np.array(env_vals)
    env_vals = env_vals.T # T x 5

    print('env params shape:', env_vals.shape)

    return env_vals


def sample_CP_random():
    CP = ControlParamSimple()

    num_days = random.randint(35, 45)
    num_hours = num_days * 24

    CP.set_endDate(num_days=num_days)
    CP.set_value("comp1.heatingpipes.pipe1.@maxTemp", 60)
    CP.set_value("comp1.heatingpipes.pipe1.@minTemp", 0)
    CP.set_value("comp1.heatingpipes.pipe1.@radiationInfluence", "0 0")

    def sample_bool():
        return random.choice([True, False])
    
    def sample_int(low, high):
        return random.randint(low, high)

    def sample_material():
        return random.choice(['scr_Transparent.par', 'scr_Shade.par', 'scr_Blackout.par'])

    def sample_line(r_x1, r_y1, r_x2, r_y2):
        x1 = sample_int(*r_x1)
        x2 = sample_int(*r_x2)
        y1 = sample_int(*r_y1)
        y2 = sample_int(*r_y2)
        return f"{x1} {y1}; {x2} {y2}"

    def sample_doseCap(r_y1, r_y2, r_y3):
        x1 = sample_int(0, 33)
        x2 = sample_int(34, 66)
        x3 = sample_int(67, 100)
        y1 = sample_int(*r_y1)
        y2 = sample_int(*r_y2)
        y3 = sample_int(*r_y3)
        return f"{x1} {y1}; {x2} {y2}; {x3} {y3}"
    
    def val_seq(func, args_val, num_steps):
        return [func(*args_val) for _ in range(num_steps)]

    def sample_screen(CP, id):
        key_prefix = f"comp1.screens.scr{id}"
        CP.set_value(f"{key_prefix}.@enabled", sample_bool())
        CP.set_value(f"{key_prefix}.@material", sample_material())
        CP.set_value(f"{key_prefix}.@ToutMax", val_seq(sample_int, (-20, 30), num_hours))
        CP.set_value(f"{key_prefix}.@closeBelow", val_seq(sample_line, [(0,10), (50,150), (10,30), (0,50)], num_hours))
        CP.set_value(f"{key_prefix}.@closeAbove", val_seq(sample_int, (800, 1400), num_hours))
        CP.set_value(f"{key_prefix}.@lightPollutionPrevention", True)
    
    # ============== sample temp params ============== 
    CP.set_value("comp1.setpoints.temp.@heatingTemp", val_seq(sample_int, (5,30), num_hours))
    CP.set_value("comp1.setpoints.temp.@ventOffset", val_seq(sample_int, (0,5), num_hours))
    CP.set_value("comp1.setpoints.temp.@radiationInfluence", "0")
    CP.set_value("comp1.setpoints.temp.@PbandVent", val_seq(sample_line, [(0,5), (10,20), (20,25), (5,10)], num_hours))
    CP.set_value("comp1.setpoints.temp.@startWnd", val_seq(sample_int, (0,50), num_hours))
    CP.set_value("comp1.setpoints.temp.@startWnd", val_seq(sample_int, (0,50), num_hours))
    CP.set_value("comp1.setpoints.temp.@winLeeMin", 0)
    CP.set_value("comp1.setpoints.temp.@winLeeMax", 100)
    CP.set_value("comp1.setpoints.temp.@winWndMin", 0)
    CP.set_value("comp1.setpoints.temp.@winWndMax", 100)
    
    # ============== sample CO2 params ============== 
    CP.set_value("common.CO2dosing.@pureCO2cap", sample_int(100, 200))
    CP.set_value("comp1.setpoints.CO2.@setpoint", val_seq(sample_int, (400, 1200), num_hours))
    CP.set_value("comp1.setpoints.CO2.@setpIfLamps", val_seq(sample_int, (400, 1200), num_hours))
    CP.set_value("comp1.setpoints.CO2.@doseCapacity", val_seq(sample_doseCap, [(70,100), (40,70), (0,40)], num_hours))
    
    # ============== sample screen params ============== 
    sample_screen(CP, 1)
    sample_screen(CP, 2)
    
    # ============== sample illumination params ==============
    CP.set_value("comp1.illumination.lmp1.@enabled", True)
    CP.set_value("comp1.illumination.lmp1.@intensity", val_seq(sample_int, (50, 200), num_hours))
    CP.set_hoursLight(val_seq(sample_int, (0, 20), num_days))
    CP.set_value("comp1.illumination.lmp1.@endTime", 20)
    CP.set_maxIglob(val_seq(sample_int, (200, 400), num_days))
    CP.set_maxPARsum(val_seq(sample_int, (10, 50), num_days))

    return CP


if __name__ == '__main__':
    import argparse

    # to prepare EP_PATH and INIT_STATE_PATH
    # (1) change COMMON_DATA_DIR in constant.py
    # (2) $ python --get-op1-pool --get-norm-data --get-ep-ndays 60 --data-dirs DIR_TO_DATA{1,2,3,...} --simulator [A|B|C|D]

    parser = argparse.ArgumentParser()
    parser.add_argument('--get-op1-pool', action='store_true')
    parser.add_argument('--get-norm-data', action='store_true')
    parser.add_argument('--get-ep-ndays', type=int, default=60)
    parser.add_argument('--data-dirs', type=str, nargs='+', default=None)
    parser.add_argument('--simulator', type=str, default='A')
    args = parser.parse_args()

    os.makedirs(COMMON_DATA_DIR, exist_ok=True)

    if args.get_op1_pool:
        op1_pool = get_op1_pool(args.data_dirs)
        np.save(INIT_STATE_PATHS[args.simulator], op1_pool)

    if args.get_ep_ndays:
        ep = get_ep_ndays(args.simulator, num_days=args.get_ep_ndays)
        np.save(EP_PATHS[args.simulator], ep)

    if args.get_norm_data:
        (cp_mean, cp_std), (ep_mean, ep_std), (op_mean, op_std) = compute_mean_std(args.data_dirs)
        np.savez_compressed(NORM_DATA_PATHS[args.simulator], cp_mean=cp_mean, cp_std=cp_std, ep_mean=ep_mean, ep_std=ep_std, op_mean=op_mean, op_std=op_std)
        