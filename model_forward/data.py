import os
import json
import inspect
import requests
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from control_param import ControlParamSimple
from constant import COMMON_DATA_DIR, ENV_KEYS, KEYS, OUTPUT_KEYS, START_DATE, URL, EP_PATHS, INIT_STATE_PATHS, OUTPUT_KEYS_TO_INDEX


DEBUG = False


def print_current_function_name():
    if DEBUG:
        print(inspect.stack()[1][3], end=', ')


def print_arr_shape(arr):
    if DEBUG:
        print(arr.shape)


def save_json_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


class ParseControl(object):
    def __init__(self, control, start_date=START_DATE):
        super().__init__()
        self.start_date = start_date
        self.end_date = datetime.date.fromisoformat(control['simset']['@endDate'])
        assert self.end_date > self.start_date
        self.num_days = (self.end_date - self.start_date).days
        self.num_hours = self.num_days * 24

    def parse(self, control):
        cp_list = [
            self.parse_pipe_maxTemp(control),
            self.parse_pipe_minTemp(control),
            self.parse_pipe_radInf(control),
            self.parse_temp_heatingTemp(control),
            self.parse_temp_ventOffset(control),
            self.parse_temp_radInf(control),
            self.parse_temp_PbandVent(control),
            self.parse_vent_startWnd(control),
            self.parse_vent_winLeeMin(control),
            self.parse_vent_winLeeMax(control),
            self.parse_vent_winWndMin(control),
            self.parse_vent_winWndMax(control),
            self.parse_co2_pureCap(control),
            self.parse_co2_setpoint(control),
            self.parse_co2_setpIfLamps(control),
            self.parse_co2_doseCap(control),
            self.parse_scr_enabled(control, 1),
            self.parse_scr_material(control, 1),
            self.parse_scr_ToutMax(control, 1),
            self.parse_scr_closeBelow(control, 1),
            self.parse_scr_closeAbove(control, 1),
            self.parse_scr_LPP(control, 1),
            self.parse_scr_enabled(control, 2),
            self.parse_scr_material(control, 2),
            self.parse_scr_ToutMax(control, 2),
            self.parse_scr_closeBelow(control, 2),
            self.parse_scr_closeAbove(control, 2),
            self.parse_scr_LPP(control, 2),
            self.parse_lmp1_enabled(control),
            self.parse_lmp1_intensity(control),
            self.parse_lmp1_hoursLight(control),
            self.parse_lmp1_endTime(control),
            self.parse_lmp1_maxIglob(control),
            self.parse_lmp1_maxPARsum(control),
            self.parse_plant_density(control),
        ]
        cp_arr = np.concatenate(cp_list, axis=0).T
        print_arr_shape(cp_arr)
        assert cp_arr.shape[1] == 56
        return cp_arr

    @staticmethod
    def get_value(control, key_path):
        value = control        
        for key in key_path.split('.'):
            value = value[key]
        return value

    @staticmethod
    def flatten(t):
        return [item for sublist in t for item in sublist]

    def value2arr(self, value, preprocess, valid_dtype):
        if type(value) in valid_dtype:
            cp_arr = [value] * self.num_hours # 1 x num_hours
        elif type(value) == dict:
            cp_arr = []
            for day_value in value.values():
                if type(day_value) in valid_dtype:
                    cp_arr.extend([day_value] * 24)
                elif type(day_value) == dict:
                    cp_arr.extend(list(day_value.values()))
        assert len(cp_arr) == self.num_hours
        cp_arr = np.asarray(list(map(preprocess, cp_arr))).T  # N x num_hours
        print_arr_shape(cp_arr)
        return cp_arr
    
    def parse_pipe_maxTemp(self, control):
        value = self.get_value(control, "comp1.heatingpipes.pipe1.@maxTemp")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_pipe_minTemp(self, control):
        value = self.get_value(control, "comp1.heatingpipes.pipe1.@minTemp")
        preprocess = lambda x: [x]
        valid_dtype = [int, float]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_pipe_radInf(self, control):
        value = self.get_value(control, "comp1.heatingpipes.pipe1.@radiationInfluence")
        valid_dtype = [str]
        
        def preprocess(s):
            numbers = [float(x) for x in s.split()]
            if len(numbers) == 1:
                numbers *= 2
            return numbers
        
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_temp_heatingTemp(self, control):
        value = self.get_value(control, "comp1.setpoints.temp.@heatingTemp")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_temp_ventOffset(self, control):
        value = self.get_value(control, "comp1.setpoints.temp.@ventOffset")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_temp_radInf(self, control):
        value = self.get_value(control, "comp1.setpoints.temp.@radiationInfluence")
        valid_dtype = [str]
        
        def preprocess(s):
            numbers = [float(x) for x in s.split()]
            if len(numbers) == 1:
                assert numbers[0] == 0
                numbers *= 3
            return numbers
        
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_temp_PbandVent(self, control):
        value = self.get_value(control, "comp1.setpoints.temp.@PbandVent")
        valid_dtype = [str]
        
        def preprocess(s):
            numbers = [[float(x) for x in y.split()] for y in s.split(';')]
            return self.flatten(numbers)

        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)

    def parse_vent_startWnd(self, control):
        value = self.get_value(control, "comp1.setpoints.ventilation.@startWnd")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_vent_winLeeMin(self, control):
        value = self.get_value(control, "comp1.setpoints.ventilation.@winLeeMin")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_vent_winLeeMax(self, control):
        value = self.get_value(control, "comp1.setpoints.ventilation.@winLeeMax")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_vent_winWndMin(self, control):
        value = self.get_value(control, "comp1.setpoints.ventilation.@winWndMin")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_vent_winWndMax(self, control):
        value = self.get_value(control, "comp1.setpoints.ventilation.@winWndMax")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_co2_pureCap(self, control):
        value = self.get_value(control, "common.CO2dosing.@pureCO2cap")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_co2_setpoint(self, control):
        value = self.get_value(control, "comp1.setpoints.CO2.@setpoint")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_co2_setpIfLamps(self, control):
        value = self.get_value(control, "comp1.setpoints.CO2.@setpIfLamps")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_co2_doseCap(self, control):
        value = self.get_value(control, "comp1.setpoints.CO2.@doseCapacity")
        valid_dtype = [str]
        
        def preprocess(s):
            if ';' in s:
                numbers = [[float(x) for x in y.split()] for y in s.split(';')]
            else:
                val = float(s)
                numbers = [[25, val], [50, val], [75, val]]
            return self.flatten(numbers)

        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)

    def parse_scr_enabled(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@enabled")
        valid_dtype = [bool]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_scr_material(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@material")
        valid_dtype = [str]

        def preprocess(s):
            choices = ['scr_Transparent.par', 'scr_Shade.par', 'scr_Blackout.par']
            return [float(s==c) for c in choices]
        
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_scr_ToutMax(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@ToutMax")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_scr_closeBelow(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@closeBelow")
        valid_dtype = [int, float, str]
        
        def preprocess(s):
            if type(s) == str:
                numbers = [[float(x) for x in y.split()] for y in s.split(';')]
            else:
                val = float(s)
                numbers = [[0, val], [10, val]]
            return self.flatten(numbers)
        
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)

    def parse_scr_closeAbove(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@closeAbove")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_scr_LPP(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@lightPollutionPrevention")
        valid_dtype = [bool]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_lmp1_enabled(self, control):
        value = self.get_value(control, "comp1.illumination.lmp1.@enabled")
        valid_dtype = [bool]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)

    def parse_lmp1_intensity(self, control):
        value = self.get_value(control, "comp1.illumination.lmp1.@intensity")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_lmp1_hoursLight(self, control):
        value = self.get_value(control, "comp1.illumination.lmp1.@hoursLight")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_lmp1_endTime(self, control):
        value = self.get_value(control, "comp1.illumination.lmp1.@endTime")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_lmp1_maxIglob(self, control):
        value = self.get_value(control, "comp1.illumination.lmp1.@maxIglob")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_lmp1_maxPARsum(self, control):
        value = self.get_value(control, "comp1.illumination.lmp1.@maxPARsum")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        print_current_function_name()
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_plant_density(self, control):
        value = self.get_value(control, "crp_lettuce.Intkam.management.@plantDensity")
        assert type(value) == str and ';' in value

        arr = np.zeros((1, self.num_hours))
        for s in value.split(';'):
            day, density = s.split()
            hour_offset = (int(day) - 1) * 24
            arr[0,hour_offset:] = float(density)

        print_current_function_name()
        print_arr_shape(arr)
        return arr


def parse_control(control):
    return ParseControl(control).parse(control)


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

    print(f'preprocessing data @ {data_dir} ...')


    OP_NON_PLANT_KEYS = [
        'comp1.Air.T',
        'comp1.Air.RH',
        'comp1.Air.ppm',
        'comp1.PARsensor.Above',
        'comp1.Lmp1.ElecUse',
        'comp1.PConPipe1.Value',
        'comp1.McPureAir.Value',
        'comp1.TPipe1.Value',
        'comp1.ConPipes.TSupPipe1',
        'comp1.ConWin.WinLee',
        'comp1.ConWin.WinWnd',
        'comp1.Setpoints.SpHeat',
        'comp1.Setpoints.SpVent',
        'comp1.Scr1.Pos',
        'comp1.Scr2.Pos',
    ]
    OP_NON_PLANT_INDEX = [OUTPUT_KEYS_TO_INDEX[key] for key in OP_NON_PLANT_KEYS]

    cp_arr, ep_arr, op_pre_arr, op_cur_arr = [], [], [], []
    for name in tqdm(names):
        control = load_json_data(os.path.join(control_dir, name))
        output = load_json_data(os.path.join(output_dir, name))

        assert output['responsemsg'] == 'ok'

        control_vals = parse_control(control) # control_vals: T x M
        env_vals, output_vals = parse_output(output) # env_vals: T x N1, output_vals: T x N2

        # assumption: cp[t] + ep[t-1] + op[t-1] -> op[t]
        cp_vals_t = control_vals[1:]
        ep_vals_t_minus_1 = env_vals[:-1]
        op_vals_t_minus_1 = output_vals[:-1, OP_NON_PLANT_INDEX]
        op_vals_t = output_vals[1:, OP_NON_PLANT_INDEX]

        assert cp_vals_t.shape[0] == ep_vals_t_minus_1.shape[0] == op_vals_t_minus_1.shape[0] == op_vals_t.shape[0]

        cp_arr.append(cp_vals_t)
        ep_arr.append(ep_vals_t_minus_1)
        op_pre_arr.append(op_vals_t_minus_1)
        op_cur_arr.append(op_vals_t)
    
    cp_arr = np.concatenate(cp_arr, axis=0)
    ep_arr = np.concatenate(ep_arr, axis=0)
    op_pre_arr = np.concatenate(op_pre_arr, axis=0)
    op_cur_arr = np.concatenate(op_cur_arr, axis=0)

    np.savez_compressed(f'{data_dir}/processed_data.npz', 
        cp=cp_arr, ep=ep_arr, op_pre=op_pre_arr, op_cur=op_cur_arr)


def preprocess_data_plant(data_dir):
    control_dir = os.path.join(data_dir, 'controls')
    output_dir = os.path.join(data_dir, 'outputs')

    assert os.path.exists(control_dir)
    assert os.path.exists(output_dir)

    cp_names = os.listdir(control_dir)
    op_names = os.listdir(output_dir)
    names = list(set(cp_names).intersection(set(op_names)))

    print(f'preprocessing data @ {data_dir} ...')

    OP_PLANT_KEYS = [
        'comp1.Plant.headFW',
        'comp1.Plant.shootDryMatterContent',
        'comp1.Plant.fractionGroundCover',
        'comp1.Plant.plantProjection',
    ]
    OP_OTHER_KEYS = [
        'comp1.Air.T',
        'comp1.Air.RH',
        'comp1.Air.ppm',
        'comp1.PARsensor.Above',
    ]
    OP_PLANT_INDEX = [OUTPUT_KEYS_TO_INDEX[key] for key in OP_PLANT_KEYS]
    OP_OTHER_INDEX = [OUTPUT_KEYS_TO_INDEX[key] for key in OP_OTHER_KEYS]

    cp_arr, ep_arr, op_other_arr, op_plant_pre_arr, op_plant_cur_arr = [], [], [], [], []
    for name in tqdm(names):
        control = load_json_data(os.path.join(control_dir, name))
        output = load_json_data(os.path.join(output_dir, name))

        assert output['responsemsg'] == 'ok'

        control_vals = parse_control(control) # control_vals: T x M
        env_vals, output_vals = parse_output(output) # env_vals: T x N1, output_vals: T x N2

        # assumption: cp[d-1] + ep[d-1] + op_other[d-1] + op_plant[d-1] -> op_plant[d]
        cp_vals = control_vals.reshape(-1, 24, control_vals.shape[1])
        ep_vals = env_vals.reshape(-1, 24, env_vals.shape[1])
        op_other_vals = output_vals[:, OP_OTHER_INDEX].reshape(-1, 24, len(OP_OTHER_INDEX))
        op_plant_vals = output_vals[:, OP_PLANT_INDEX].reshape(-1, 24, len(OP_PLANT_INDEX))
        op_plant_vals = np.mean(op_plant_vals, axis=1)
        
        cp_vals = cp_vals[:-1]
        ep_vals = ep_vals[:-1]
        op_other_vals = op_other_vals[:-1]
        op_plant_pre_vals = op_plant_vals[:-1]
        op_plant_cur_vals = op_plant_vals[1:]

        cp_arr.append(cp_vals)
        ep_arr.append(ep_vals)
        op_other_arr.append(op_other_vals)
        op_plant_pre_arr.append(op_plant_pre_vals)
        op_plant_cur_arr.append(op_plant_cur_vals)
    
    cp_arr = np.concatenate(cp_arr, axis=0)
    ep_arr = np.concatenate(ep_arr, axis=0)
    op_other_arr = np.concatenate(op_other_arr, axis=0)
    op_plant_pre_arr = np.concatenate(op_plant_pre_arr, axis=0)
    op_plant_cur_arr = np.concatenate(op_plant_cur_arr, axis=0)

    np.savez_compressed(f'{data_dir}/processed_data_plant.npz', 
        cp=cp_arr, ep=ep_arr, op_other=op_other_arr, op_plant_pre=op_plant_pre_arr, op_plant_cur=op_plant_cur_arr)


def zscore_normalize(data_arr, mean_arr, std_arr):
    std_arr[std_arr==0] = 1
    norm_arr = (data_arr - mean_arr) / std_arr
    return np.nan_to_num(norm_arr)


def zscore_denormalize(norm_arr, mean_arr, std_arr):
    return norm_arr * std_arr + mean_arr


def compute_mean_std(data_dirs):
    cp_all, ep_all, op_all = [], [], []
    
    for data_dir in data_dirs:
        proc_data = np.load(f'{data_dir}/processed_data.npz')
        cp_all.append(proc_data['cp'])
        ep_all.append(proc_data['ep'])
        op_all.append(proc_data['op_pre'])
    
    cp_all = np.concatenate(cp_all, axis=0)
    ep_all = np.concatenate(ep_all, axis=0)
    op_all = np.concatenate(op_all, axis=0)

    def get_mean_std(arr):
        return np.mean(arr, axis=0, dtype=np.float32), np.std(arr, axis=0, dtype=np.float32)

    cp_mean, cp_std = get_mean_std(cp_all)
    ep_mean, ep_std = get_mean_std(ep_all)
    op_mean, op_std = get_mean_std(op_all)

    return {
        'cp_mean': cp_mean, 'cp_std': cp_std,
        'ep_mean': ep_mean, 'ep_std': ep_std,
        'op_mean': op_mean, 'op_std': op_std,
    }


def compute_mean_std_plant(data_dirs):
    cp_all, ep_all, op_other_all, op_plant_all = [], [], [], []
    
    for data_dir in data_dirs:
        proc_data = np.load(f'{data_dir}/processed_data_plant.npz')
        cp_all.append(proc_data['cp'])
        ep_all.append(proc_data['ep'])
        op_other_all.append(proc_data['op_other'])
        op_plant_all.append(proc_data['op_plant_pre'])
    
    cp_all = np.concatenate(cp_all, axis=0)
    ep_all = np.concatenate(ep_all, axis=0)
    op_other_all = np.concatenate(op_other_all, axis=0)
    op_plant_all = np.concatenate(op_plant_all, axis=0)

    def get_mean_std(arr):
        return np.mean(arr, axis=0, dtype=np.float32), np.std(arr, axis=0, dtype=np.float32)

    cp_mean, cp_std = get_mean_std(cp_all)
    ep_mean, ep_std = get_mean_std(ep_all)
    op_other_mean, op_other_std = get_mean_std(op_other_all)
    op_plant_mean, op_plant_std = get_mean_std(op_plant_all)

    return {
        'cp_mean': cp_mean, 'cp_std': cp_std,
        'ep_mean': ep_mean, 'ep_std': ep_std,
        'op_other_mean': op_other_mean, 'op_other_std': op_other_std,
        'op_plant_mean': op_plant_mean, 'op_plant_std': op_plant_std,
    }


class AGCDataset(Dataset):
    def __init__(self, data_dirs, norm_data=None):
        super(AGCDataset, self).__init__()

        cp, ep, op_pre, op_cur = [], [], [], []
        
        for data_dir in data_dirs:
            data_path = f'{data_dir}/processed_data.npz'
            assert os.path.exists(data_path)
            data = np.load(data_path)

            cp.append(data['cp'])
            ep.append(data['ep'])
            op_pre.append(data['op_pre'])
            op_cur.append(data['op_cur'])

        self.cp = np.concatenate(cp, axis=0, dtype=np.float32)
        self.ep = np.concatenate(ep, axis=0, dtype=np.float32)
        self.op_pre = np.concatenate(op_pre, axis=0, dtype=np.float32)
        self.op_cur = np.concatenate(op_cur, axis=0, dtype=np.float32)

        if norm_data:
            self.cp = zscore_normalize(self.cp, norm_data['cp_mean'], norm_data['cp_std'])
            self.ep = zscore_normalize(self.ep, norm_data['ep_mean'], norm_data['ep_std'])
            self.op_pre = zscore_normalize(self.op_pre, norm_data['op_mean'], norm_data['op_std'])
            self.op_cur = zscore_normalize(self.op_cur, norm_data['op_mean'], norm_data['op_std'])

    def __getitem__(self, index):
        return self.cp[index], self.ep[index], self.op_pre[index], self.op_cur[index]

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


class AGCDatasetPlant(Dataset):
    def __init__(self, data_dirs, norm_data=None):
        super(AGCDatasetPlant, self).__init__()

        cp, ep, op_other, op_plant_pre, op_plant_cur = [], [], [], [], []
        
        for data_dir in data_dirs:
            data_path = f'{data_dir}/processed_data_plant.npz'
            assert os.path.exists(data_path)
            data = np.load(data_path)

            cp.append(data['cp'])
            ep.append(data['ep'])
            op_other.append(data['op_other'])
            op_plant_pre.append(data['op_plant_pre'])
            op_plant_cur.append(data['op_plant_cur'])

        self.cp = np.concatenate(cp, axis=0, dtype=np.float32)
        self.ep = np.concatenate(ep, axis=0, dtype=np.float32)
        self.op_other = np.concatenate(op_other, axis=0, dtype=np.float32)
        self.op_plant_pre = np.concatenate(op_plant_pre, axis=0, dtype=np.float32)
        self.op_plant_cur = np.concatenate(op_plant_cur, axis=0, dtype=np.float32)

        if norm_data:
            self.cp = zscore_normalize(self.cp, norm_data['cp_mean'], norm_data['cp_std'])
            self.ep = zscore_normalize(self.ep, norm_data['ep_mean'], norm_data['ep_std'])
            self.op_other = zscore_normalize(self.op_other, norm_data['op_other_mean'], norm_data['op_other_std'])
            self.op_plant_pre = zscore_normalize(self.op_plant_pre, norm_data['op_plant_mean'], norm_data['op_plant_std'])
            self.op_plant_cur = zscore_normalize(self.op_plant_cur, norm_data['op_plant_mean'], norm_data['op_plant_std'])

    def __getitem__(self, index):
        return self.cp[index], self.ep[index], self.op_other[index], self.op_plant_pre[index], self.op_plant_cur[index]

    def __len__(self):
        return self.cp.shape[0]

    @property
    def cp_dim(self):
        return self.cp.shape[2]
    
    @property
    def ep_dim(self):
        return self.ep.shape[2]
    
    @property
    def op_other_dim(self):
        return self.op_other.shape[2]
    
    @property
    def op_plant_dim(self):
        return self.op_plant_pre.shape[2]


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
    # (2) $ python --get-op1-pool --get-ep-ndays 60 --data-dirs DIR_TO_DATA{1,2,3,...} --simulator [A|B|C|D]

    parser = argparse.ArgumentParser()
    parser.add_argument('--get-op1-pool', action='store_true')
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
    