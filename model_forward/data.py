import os
import json
import datetime
import numpy as np

from astral.geocoder import lookup, database
from astral.sun import sun
from torch.utils.data import Dataset


CITY = lookup('Amsterdam', database())
START_DATE = datetime.date(2021, 3, 4)

CONTROL_KEYS = [
    "simset.@endDate",
    "comp1.heatingpipes.pipe1.@maxTemp",
    "comp1.heatingpipes.pipe1.@minTemp",
    "comp1.heatingpipes.pipe1.@radiationInfluence",
    "comp1.setpoints.temp.@heatingTemp",
    "comp1.setpoints.temp.@ventOffset",
    "comp1.setpoints.temp.@radiationInfluence",
    "comp1.setpoints.temp.@PbandVent",
    "comp1.setpoints.ventilation.@startWnd",
    "comp1.setpoints.ventilation.@winLeeMin",
    "comp1.setpoints.ventilation.@winLeeMax",
    "comp1.setpoints.ventilation.@winWndMin",
    "comp1.setpoints.ventilation.@winWndMax",
    "common.CO2dosing.@pureCO2cap",
    "comp1.setpoints.CO2.@setpoint",
    "comp1.setpoints.CO2.@setpIfLamps",
    "comp1.setpoints.CO2.@doseCapacity",
    "comp1.screens.scr1.@enabled",
    "comp1.screens.scr1.@material",
    "comp1.screens.scr1.@ToutMax",
    "comp1.screens.scr1.@closeBelow",
    "comp1.screens.scr1.@closeAbove",
    "comp1.screens.scr1.@lightPollutionPrevention",
    "comp1.screens.scr2.@enabled",
    "comp1.screens.scr2.@material",
    "comp1.screens.scr2.@ToutMax",
    "comp1.screens.scr2.@closeBelow",
    "comp1.screens.scr2.@closeAbove",
    "comp1.screens.scr2.@lightPollutionPrevention",
    "comp1.illumination.lmp1.@enabled",
    "comp1.illumination.lmp1.@intensity",
    "comp1.illumination.lmp1.@hoursLight",
    "comp1.illumination.lmp1.@endTime",
    "comp1.illumination.lmp1.@maxIglob",
    "comp1.illumination.lmp1.@maxPARsum",
    "crp_lettuce.Intkam.management.@plantDensity",
]

ENV_KEYS = [
    'common.Iglob.Value',
    'common.TOut.Value',
    'common.RHOut.Value',
    'common.Windsp.Value',
    'common.Economics.PeakHour',
]

OUTPUT_KEYS = [
    'comp1.Air.T',
    'comp1.Air.RH',
    'comp1.Air.ppm',
    'comp1.PARsensor.Above',
    'comp1.TPipe1.Value',
    'comp1.ConPipes.TSupPipe1',
    'comp1.PConPipe1.Value',
    'comp1.ConWin.WinLee',
    'comp1.ConWin.WinWnd',
    'comp1.Setpoints.SpHeat',
    'comp1.Setpoints.SpVent',
    'comp1.Scr1.Pos',
    'comp1.Scr2.Pos',
    'comp1.McPureAir.Value',
    'comp1.Plant.headFW',
    'comp1.Plant.fractionGroundCover',
    'comp1.Plant.shootDryMatterContent',
    'comp1.Plant.plantProjection',
    'comp1.Plant.PlantDensity',
    'comp1.Lmp1.ElecUse',
]


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


def preprocess_data(data_dir):
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
        cp, ep, op = preprocess_data(data_dir)
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
    def __init__(self, data_dirs, transform=None):
        super(SupervisedModelDataset, self).__init__()

        self.transform = transform

        cp_arr_list, ep_arr_list = [], []
        op_pre_arr_list, op_cur_arr_list = [], []
        
        for data_dir in data_dirs:
            data_path = os.path.join(data_dir, 'processed_data.npz')
            
            if not os.path.exists(data_path):
                preprocess_data(data_dir)

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

    def __getitem__(self, index):
        cp_vec = self.cp[index]
        ep_vec = self.ep[index]
        op_pre_vec = self.op_pre[index]
        op_cur_vec = self.op_cur[index]

        if self.transform:
            cp_vec, ep_vec, op_pre_vec, op_cur_vec = self.transform(cp_vec, ep_vec, op_pre_vec, op_cur_vec)

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
    