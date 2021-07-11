import os
import datetime
import numpy as np

from torch.utils.data import Dataset

from tqdm import tqdm
from astral.sun import sun
from astral.geocoder import lookup, database
from scipy.interpolate import interp1d

from constant import START_DATE, CITY_NAME, ENV_KEYS, OUTPUT_KEYS
from utils import load_json_data


class ParseControl(object):
    def __init__(self, start_date, city_name):
        super(ParseControl, self).__init__()
        self.city = lookup(city_name, database())
        self.start_date = start_date

    def parse(self, control):
        end_date = datetime.date.fromisoformat(control['simset']['@endDate'])
        assert end_date > self.start_date
        num_days = (end_date - self.start_date).days
        self.num_hours = num_days * 24
        
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
        return np.concatenate(cp_list, axis=0, dtype=np.float32).T

    @staticmethod
    def get_value(control, key_path):
        value = control        
        for key in key_path.split('.'):
            value = value[key]
        return value

    @staticmethod
    def flatten(t):
        return [item for sublist in t for item in sublist]

    @staticmethod
    def get_sun_rise_and_set(dateinfo, cityinfo):
        s = sun(cityinfo.observer, date=dateinfo, tzinfo=cityinfo.timezone)
        h_r = s['sunrise'].hour + s['sunrise'].minute / 60
        h_s = s['sunset'].hour + s['sunset'].minute / 60
        return h_r, h_s

    def value2arr(self, value, preprocess, valid_dtype, interpolate='previous'):
        if type(value) in valid_dtype:
            cp_arr = [preprocess(value)] * self.num_hours # 1 x num_hours
        elif type(value) == dict:
            cp_arr = self.dict_value2arr(value, valid_dtype, preprocess, interpolate)
        return np.asarray(cp_arr).T  # N x num_hours

    def dict_value2arr(self, value, valid_dtype, preprocess, interpolate):
        result = [None] * self.num_hours
        for date, day_value in value.items():
            day, month = date.split('-')
            day, month = int(day), int(month)
            dateinfo = datetime.date(2021, month, day)
            day_offset = (dateinfo - self.start_date).days
            h_start = day_offset * 24
            r, s = self.get_sun_rise_and_set(dateinfo, self.city)
            
            if type(day_value) in valid_dtype:
                result[h_start:h_start+24] = [preprocess(day_value)] * 24
            elif type(day_value) == dict:
                cp_arr_day = [None] * 24
                for hour, hour_value in day_value.items():
                    h_cur = round(eval(hour, {'r': r, 's': s}))
                    cp_arr_day[h_cur] = preprocess(hour_value)
                result[h_start:h_start+24] = cp_arr_day
            else:
                raise ValueError

        # interpolate missing time hours
        points = [(i,v) for i, v in enumerate(result) if v is not None]
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        if x[0] > 0:
            x.insert(0, 0)
            y.insert(0, y[0])
        if x[-1] < self.num_hours-1:
            x.append(self.num_hours-1)
            y.append(y[-1])
        f = interp1d(x, y, kind=interpolate, axis=0)

        return f(range(self.num_hours))
    
    def parse_pipe_maxTemp(self, control):
        value = self.get_value(control, "comp1.heatingpipes.pipe1.@maxTemp")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype, interpolate='linear')
    
    def parse_pipe_minTemp(self, control):
        value = self.get_value(control, "comp1.heatingpipes.pipe1.@minTemp")
        preprocess = lambda x: [x]
        valid_dtype = [int, float]
        return self.value2arr(value, preprocess, valid_dtype, interpolate='linear')
    
    def parse_pipe_radInf(self, control):
        value = self.get_value(control, "comp1.heatingpipes.pipe1.@radiationInfluence")
        valid_dtype = [str]
        def preprocess(s):
            numbers = [float(x) for x in s.split()]
            if len(numbers) == 1:
                numbers *= 2
            return numbers
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_temp_heatingTemp(self, control):
        value = self.get_value(control, "comp1.setpoints.temp.@heatingTemp")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype, interpolate='linear')
    
    def parse_temp_ventOffset(self, control):
        value = self.get_value(control, "comp1.setpoints.temp.@ventOffset")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype, interpolate='linear')
    
    def parse_temp_radInf(self, control):
        value = self.get_value(control, "comp1.setpoints.temp.@radiationInfluence")
        valid_dtype = [str]
        def preprocess(s):
            numbers = [float(x) for x in s.split()]
            if len(numbers) == 1:
                assert numbers[0] == 0
                numbers *= 3
            return numbers
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_temp_PbandVent(self, control):
        value = self.get_value(control, "comp1.setpoints.temp.@PbandVent")
        valid_dtype = [str]
        def preprocess(s):
            numbers = [[float(x) for x in y.split()] for y in s.split(';')]
            return self.flatten(numbers)
        return self.value2arr(value, preprocess, valid_dtype)

    def parse_vent_startWnd(self, control):
        value = self.get_value(control, "comp1.setpoints.ventilation.@startWnd")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_vent_winLeeMin(self, control):
        value = self.get_value(control, "comp1.setpoints.ventilation.@winLeeMin")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_vent_winLeeMax(self, control):
        value = self.get_value(control, "comp1.setpoints.ventilation.@winLeeMax")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_vent_winWndMin(self, control):
        value = self.get_value(control, "comp1.setpoints.ventilation.@winWndMin")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_vent_winWndMax(self, control):
        value = self.get_value(control, "comp1.setpoints.ventilation.@winWndMax")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_co2_pureCap(self, control):
        value = self.get_value(control, "common.CO2dosing.@pureCO2cap")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_co2_setpoint(self, control):
        value = self.get_value(control, "comp1.setpoints.CO2.@setpoint")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_co2_setpIfLamps(self, control):
        value = self.get_value(control, "comp1.setpoints.CO2.@setpIfLamps")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
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
        return self.value2arr(value, preprocess, valid_dtype)

    def parse_scr_enabled(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@enabled")
        valid_dtype = [bool]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_scr_material(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@material")
        valid_dtype = [str]
        def preprocess(s):
            choices = ['scr_Transparent.par', 'scr_Shade.par', 'scr_Blackout.par']
            return [float(s==c) for c in choices]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_scr_ToutMax(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@ToutMax")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
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
        return self.value2arr(value, preprocess, valid_dtype)

    def parse_scr_closeAbove(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@closeAbove")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_scr_LPP(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@lightPollutionPrevention")
        valid_dtype = [bool]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_lmp1_enabled(self, control):
        value = self.get_value(control, "comp1.illumination.lmp1.@enabled")
        valid_dtype = [bool]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)

    def parse_lmp1_intensity(self, control):
        value = self.get_value(control, "comp1.illumination.lmp1.@intensity")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_lmp1_hoursLight(self, control):
        value = self.get_value(control, "comp1.illumination.lmp1.@hoursLight")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_lmp1_endTime(self, control):
        value = self.get_value(control, "comp1.illumination.lmp1.@endTime")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_lmp1_maxIglob(self, control):
        value = self.get_value(control, "comp1.illumination.lmp1.@maxIglob")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_lmp1_maxPARsum(self, control):
        value = self.get_value(control, "comp1.illumination.lmp1.@maxPARsum")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)
    
    def parse_plant_density(self, control):
        value = self.get_value(control, "crp_lettuce.Intkam.management.@plantDensity")
        setpoints = self.parse_plant_density_to_setpoints(value)
        arr = np.zeros((1, self.num_hours))
        for day, density in setpoints:
            hour_offset = (day - 1) * 24
            if hour_offset >= self.num_hours:
                break
            arr[0,hour_offset:] = density
        return arr

    @staticmethod
    def parse_plant_density_to_setpoints(pd_str):
        assert type(pd_str) == str
        day_density_strs = pd_str.strip().split(';')
        
        setpoints = []
        for day_density in day_density_strs:
            try:
                day, density = day_density.split()
                day, density = int(day), float(density)
            except:
                raise ValueError(f'"{day_density}" has invalid format or data types')
            
            assert day >= 1
            assert 1 <= density <= 90
            setpoints.append((day, density))
        
        days = [sp[0] for sp in setpoints]
        densities = [sp[1] for sp in setpoints]
        assert days[0] == 1  # must start with the 1st day
        assert sorted(days) == days  # days must be ascending
        assert sorted(densities, reverse=True) == densities  # densities must be descending
        
        return setpoints


def parse_control(control, start_date=START_DATE, city_name=CITY_NAME):
    return ParseControl(start_date, city_name).parse(control)


def parse_output_to_OP(output, keys):
    output_vals = []
    for key in keys:
        val = output['data'][key]['data']
        val = [0 if x == 'NaN' else x for x in val]
        output_vals.append(val)
    output_vals = np.asarray(output_vals, dtype=np.float32)
    return output_vals.T # T x OP_DIM


def parse_output_to_EP(output, keys):
    env_vals = []
    for key in keys:
        val = output['data'][key]['data']
        env_vals.append(val)
    env_vals = np.asarray(env_vals, dtype=np.float32)
    return env_vals.T # T x EP_DIM


def preprocess_data(data_dir, save_name, ep_keys, op_keys):
    control_dir = os.path.join(data_dir, 'controls')
    output_dir = os.path.join(data_dir, 'outputs')

    assert os.path.exists(control_dir)
    assert os.path.exists(output_dir)

    cp_names = os.listdir(control_dir)
    op_names = os.listdir(output_dir)
    names = list(set(cp_names).intersection(set(op_names)))

    print(f'preprocessing data @ {data_dir} ...')

    cp_arr, ep_arr, op_pre_arr, op_cur_arr = [], [], [], []
    for name in tqdm(names):
        control = load_json_data(os.path.join(control_dir, name))
        output = load_json_data(os.path.join(output_dir, name))

        if output['responsemsg'] != 'ok':
            continue

        control_vals = parse_control(control) # control_vals: T x CP_DIM
        env_vals = parse_output_to_EP(output, ep_keys) # env_vals: T x EP_DIM
        output_vals = parse_output_to_OP(output, op_keys) # output_vals: T x OP_DIM

        # assumption: cp[t] + ep[t-1] + op[t-1] -> op[t]
        cp_vals = control_vals[1:]
        ep_vals = env_vals[:-1]
        op_vals = output_vals[:-1]
        op_next_vals = output_vals[1:]

        assert cp_vals.shape[0] == ep_vals.shape[0] == op_vals.shape[0] == op_next_vals.shape[0]

        cp_arr.append(cp_vals)
        ep_arr.append(ep_vals)
        op_pre_arr.append(op_vals)
        op_cur_arr.append(op_next_vals)
    
    cp_arr = np.concatenate(cp_arr, axis=0)
    ep_arr = np.concatenate(ep_arr, axis=0)
    op_pre_arr = np.concatenate(op_pre_arr, axis=0)
    op_cur_arr = np.concatenate(op_cur_arr, axis=0)

    np.savez_compressed(f'{data_dir}/{save_name}.npz', cp=cp_arr, ep=ep_arr, op=op_pre_arr, op_next=op_cur_arr)


def get_norm_data(dataset):
    data = dataset.get_data()

    cp = data['cp']
    ep = data['ep']
    op = data['op']
    delta = data['op_next'] - data['op']
    
    return {
        'cp_mean': np.mean(cp, axis=0), 'cp_std': np.std(cp, axis=0),
        'ep_mean': np.mean(ep, axis=0), 'ep_std': np.std(ep, axis=0),
        'op_mean': np.mean(op, axis=0), 'op_std': np.std(op, axis=0),
        'delta_mean': np.mean(delta, axis=0), 'delta_std': np.std(delta, axis=0),
    }


class AGCDataset(Dataset):
    def __init__(self, 
        data_dirs, 
        processed_data_name='processed_data', 
        ep_keys=ENV_KEYS, 
        op_keys=OUTPUT_KEYS,
        force_preprocess=False, 
    ):
        super(AGCDataset, self).__init__()

        cp, ep, op, op_next = [], [], [], []
        for data_dir in data_dirs:
            data_path = f'{data_dir}/{processed_data_name}.npz'
            if not os.path.exists(data_path) or force_preprocess:
                preprocess_data(data_dir, processed_data_name, ep_keys, op_keys)
            data = np.load(data_path)

            cp.append(data['cp'])
            ep.append(data['ep'])
            op.append(data['op'])
            op_next.append(data['op_next'])

        self.cp = np.concatenate(cp, axis=0, dtype=np.float32)
        self.ep = np.concatenate(ep, axis=0, dtype=np.float32)
        self.op = np.concatenate(op, axis=0, dtype=np.float32)
        self.op_next = np.concatenate(op_next, axis=0, dtype=np.float32)

    def __getitem__(self, index):
        input = (self.cp[index], self.ep[index], self.op[index])
        target = self.op_next[index] 
        return input, target

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
        return self.op.shape[1]

    def get_data(self):
        return {
            'cp': self.cp, 
            'ep': self.ep, 
            'op': self.op, 
            'op_next': self.op_next
        }


def prepare_op_traces(data_dirs):
    op_traces = []
    for data_dir in data_dirs:
        output_dir = os.path.join(data_dir, 'outputs')
        print(f'extracting OP traces from {data_dir} ...')
        for name in tqdm(os.listdir(output_dir)):
            output = load_json_data(f'{output_dir}/{name}')
            op_trace = parse_output_to_OP(output, OUTPUT_KEYS)  # T x OP_DIM -> T could be different for different output files
            op_traces.append(op_trace)
    return op_traces


if __name__ == '__main__':
    # import argparse

    prefix = '/home/liuys/Greenhouse-Growing-Challenge/collect_data/data_sample/sim=A/'

    data_folders = [
        'data_sample=BO_data_2021-07-07_SIM=A_DS=DPD_OPT=gbrt_NI=500_NC=5000_P=0',
        'data_sample=BO_data_2021-07-08_SIM=A_DS=BSTPD_OPT=gbrt_NI=500_NC=5000_P=0',
        'data_sample=BO_date=0626_sim=A_method=gp_init=200_ncall=1000',
        'data_sample=BO_date=0626_sim=A_space=B_N=1000_init=100',
        'data_sample=BO_date=0627_sim=A_method=gbrt_init=100_ncall=1000_RS=1234',
        'data_sample=BO_date=0628_method=gbrt_init=1000_ncall=10000_RS=42_SP=A_P=1_sim=A',
        'data_sample=BO_date=0628_method=gbrt_init=100_ncall=1000_RS=1_SP=F_P=1_sim=A',
        'data_sample=BO_date=0628_sim=A_method=gbrt_init=100_ncall=1000_RS=12345_SP=E_P=1',
        'data_sample=BO_date=2021-06-30_SIM=A_DS=AA_OPT=gbrt_NI=500_NC=5000_P=0',
        'data_sample=BO_date=2021-07-08_SIM=A_DS=A4BSTPD2_OPT=gbrt_NI=100_NC=1000_P=0',
        'data_sample=BO_date=2021-07-08_SIM=A_DS=B3_OPT=gbrt_NI=10_NC=100_P=2',
        'data_sample=BO_date=2021-07-08_SIM=A_DS=B3_OPT=gbrt_NI=30_NC=300_P=2',
        'data_sample=BO_date=2021-07-09_SIM=A_DS=A4PD2_OPT=gbrt_NI=100_NC=1000_P=0',
        'data_sample=BO_date=2021-07-09_SIM=A_DS=A4PD2_OPT=gbrt_NI=500_NC=5000_P=0',
        'data_sample=grid_search_date=2021-06-19_sim=A_number=360',
        'data_sample=original_random_date=2021-06-19_sim=A_number=1000',
        'data_sample=original_random_date=2021-06-20_sim=A_number=2000',
        'data_sample=original_random_date=2021-06-21_sim=A_number=5000',
        'data_sample=original_random_date=2021-06-22_sim=A_number=10000',
        'data_sample=original_random_date=2021-06-23_sim=A_number=5000',
        'data_sample=original_random_date=2021-06-24_sim=A_number=5000',
        'data_sample=original_random_date=2021-06-25_sim=A_number=8869',
        'data_sample=random_date=2021-07-05_sim=A_number=100',
        'data_sample=random_date=2021-07-06_sim=A_number=1000'
    ]

    op_traces = prepare_op_traces(data_folders)
    np.save('op_traces.npy', op_traces)

    # data_dirs = ' '.join(f'{prefix}{f}' for f in data_folders)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data-dirs', type=str, nargs='+', default=data_dirs)
    # args = parser.parse_args()

    # dataset = AGCDataset(args.data_dirs, 'test_preprocess', force_preprocess=True)
    # norm_data = get_norm_data(dataset)

    # print(norm_data)

    # CP_parser = ParseControl(START_DATE, CITY_NAME)
    # sps = CP_parser.parse_plant_density_to_setpoints("  1  80; 3   70 ; 3 70; 5 60; 7 1")
    # print(sps)

    # op_traces = prepare_op_traces(args.data_dirs)
    