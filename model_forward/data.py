import os
import json
import requests
import datetime
import numpy as np

from astral.sun import sun
from torch.utils.data import Dataset

from control_param import ControlParamSimple
from constant import CITY, COMMON_DATA_DIR, CONTROL_KEYS, ENV_KEYS, KEYS, NORM_DATA_PATHS, OUTPUT_KEYS, START_DATE, URL, EP_PATHS, INIT_STATE_PATHS


def save_json_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def preprocess_table(param):
    numbers = [[float(x) for x in y.split()] for y in param.split(';')]
    return np.asarray(numbers).flatten()


def preprocess_screen_material(param):
    return param == np.asarray(['scr_Transparent.par', 'scr_Shade.par', 'scr_Blackout.par'])


def preprocess_screen_threshold(param):
    if isinstance(param, str):
        return preprocess_table(param)
    return np.asarray([0, param, 10, param])


def preprocess_plant_density(param, num_hours):
    spacing_scheme = [[int(x) for x in y.split()] for y in param.split(';')]
    densities = np.zeros(num_hours, dtype=np.int)
    for day, density in spacing_scheme:
        t = (day - 1) * 24
        densities[t:] = density
    return densities[np.newaxis,:]


def preprocess_pipe_radinf(param):
    numbers = [float(x) for x in param.split()]
    if len(numbers) == 1:
        assert numbers[0] == 0
        numbers *= 2
    return np.asarray(numbers)


def preprocess_co2_dose_capacity(param):
    if ';' in param:
        return preprocess_table(param)
    val = float(param)
    return np.asarray([25, val, 50, val, 75, val])


special_preprocess = {
    'comp1.screens.scr1.@material': preprocess_screen_material,
    'comp1.screens.scr2.@material': preprocess_screen_material,
    'comp1.screens.scr1.@closeBelow': preprocess_screen_threshold,
    'comp1.screens.scr2.@closeBelow': preprocess_screen_threshold,
    'comp1.setpoints.CO2.@doseCapacity': preprocess_co2_dose_capacity,
    'comp1.heatingpipes.pipe1.@radiationInfluence': preprocess_pipe_radinf,
}


def param_to_vec(key, param):
    if key in special_preprocess:
        vec = special_preprocess[key](param)
    elif type(param) in [int, float, bool]:
        vec = np.asarray([param])
    elif type(param) == str:
        vec = preprocess_table(param)
    else:
        raise ValueError(f'praram {key}:{param} data type not supported!')
    return vec


def parse_static_param(key, param, num_hours):
    if key == 'crp_lettuce.Intkam.management.@plantDensity':
        return preprocess_plant_density(param, num_hours)

    vec = param_to_vec(key, param)
    return np.repeat(vec[:, np.newaxis], num_hours, axis=1)


def parse_dynamic_param(key, param, num_hours):  # sourcery no-metrics
    value_scheme = [None] * num_hours
    valid_data_type = (int, float, str, bool)

    for date in param:
        day, month = [int(x) for x in date.split('-')]
        dateinfo = datetime.date(2021, month, day)
        day_offset = max(0, (dateinfo - START_DATE).days)
        day_start_hour_offset = day_offset * 24

        day_param = param[date]
        if type(day_param) in valid_data_type:
            value_scheme[day_start_hour_offset] = day_param
        elif isinstance(day_param, dict): # this is a 24-hour schedule
            s = sun(CITY.observer, date=dateinfo, tzinfo=CITY.timezone)
            t_sunrise = s['sunrise'].hour + s['sunrise'].minute / 60
            t_sunset = s['sunset'].hour + s['sunset'].minute / 60
            
            for time, value in day_param.items():
                assert type(time) == str and type(value) in valid_data_type
                time = time.replace('r', str(round(t_sunrise, 1)))
                time = time.replace('s', str(round(t_sunset, 1)))
                hour_offset = day_start_hour_offset + round(eval(time))
                value_scheme[hour_offset] = value
        else:
            raise ValueError(f'{day_param}: unexpected control param data type!')

    has_value_idxs = [i for i, value in enumerate(value_scheme) if value is not None]
    if has_value_idxs[0] > 0:
        value_scheme[0] = value_scheme[has_value_idxs[0]]
        has_value_idxs = [0] + has_value_idxs
    if has_value_idxs[-1] < num_hours-1:
        value_scheme[-1] = value_scheme[has_value_idxs[-1]]
        has_value_idxs = has_value_idxs + [num_hours-1]

    for i in range(len(has_value_idxs)-1):
        t_left = has_value_idxs[i]
        v_left = value_scheme[t_left]
        t_right = has_value_idxs[i+1]
        v_right = value_scheme[t_right]

        dt = t_right - t_left
        if type(v_left) in (int, float):  # missing time steps can be interpolated
            v_left = param_to_vec(key, v_left)
            v_right = param_to_vec(key, v_right)
            dv = v_right - v_left
            for t in range(t_left, t_right):
                value_scheme[t] = v_left + (t - t_left) * dv / dt
        else:  # missing time steps are just copied from previous setpoints
            v_left = param_to_vec(key, v_left)
            for t in range(t_left, t_right):
                value_scheme[t] = v_left

    value_scheme[-1] = param_to_vec(key, value_scheme[-1])
    return np.asarray(value_scheme, dtype=np.float32).T


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
        
        # assumption: cp[t] + ep[t-1] + op[t-1] -> op[t]
        cp_vals_t = control_vals[1:]
        ep_vals_t_minus_1 = env_vals[:-1]
        op_vals_t_minus_1 = output_vals[:-1]
        op_vals_t = output_vals[1:]

        assert cp_vals_t.shape[0] == ep_vals_t_minus_1.shape[0] == op_vals_t_minus_1.shape[0] == op_vals_t.shape[0]

        cp_array.append(cp_vals_t)
        ep_array.append(ep_vals_t_minus_1)
        op_pre_array.append(op_vals_t_minus_1)
        op_cur_array.append(op_vals_t)
    
    cp_array = np.concatenate(cp_array, axis=0)
    ep_array = np.concatenate(ep_array, axis=0)
    op_pre_array = np.concatenate(op_pre_array, axis=0)
    op_cur_array = np.concatenate(op_cur_array, axis=0)

    if save:
        np.savez_compressed(
            f'{data_dir}/processed_data.npz', 
            cp=cp_array, ep=ep_array, 
            op_pre=op_pre_array, op_cur=op_cur_array
        )

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
        # print(response, output['responsemsg'])

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
    