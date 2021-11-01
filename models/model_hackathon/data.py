from collections import OrderedDict
import datetime
import json
import numpy as np
import os

from tqdm import tqdm
from functools import partial
from astral.sun import sun
from astral.geocoder import lookup, database
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from constant import CITY_NAME, MATERIALS, PL_INIT_VALUE
from constant import CP_KEYS, EP_KEYS, OP_KEYS, OP_IN_KEYS, PL_KEYS
from constant import ACTION_PARAM_SPACE, BO_CONTROL_PATH, CONTROL_FIX
from utils import load_json_data, save_json_data, normalize, normalize_zero2one, NestedDefaultDict


class ActionParser:
    def __init__(self, action_dict, city_name=CITY_NAME):
        self.action_dict = action_dict
        start_date = action_dict['simset.@startDate']
        start_date = datetime.date.fromisoformat(start_date)
        day_offset = action_dict['day_offset']
        self.date = start_date + datetime.timedelta(days=day_offset)
        self.city = lookup(city_name, database())

        self.parser_router = {
            # RL params
            "comp1.setpoints.temp.@heatingTemp": self.parse_setpoint,
            "comp1.setpoints.temp.@ventOffset": self.parse_other,
            "comp1.setpoints.ventilation.@startWnd": self.parse_other,
            "comp1.setpoints.CO2.@setpoint": self.parse_setpoint,
            "comp1.screens.scr1.@ToutMax": self.parse_other,
            "comp1.screens.scr1.@closeBelow": self.parse_other,
            "comp1.screens.scr1.@closeAbove": self.parse_other,
            "comp1.screens.scr2.@ToutMax": self.parse_other,
            "comp1.screens.scr2.@closeBelow": self.parse_other,
            "comp1.screens.scr2.@closeAbove": self.parse_other,
            "comp1.illumination.lmp1.@hoursLight": self.parse_other,
            "comp1.illumination.lmp1.@endTime": self.parse_other,
            "crp_lettuce.Intkam.management.@plantDensity": self.parse_other,
            # BO params
            "common.CO2dosing.@pureCO2cap": self.parse_other,  # e.g. 280
            "comp1.screens.scr1.@enabled": self.parse_enable,  # e.g. True
            "comp1.screens.scr1.@material": self.parse_material,  # e.g. 'scr_Blackout.par'
            "comp1.screens.scr2.@enabled": self.parse_enable,  # e.g. False
            "comp1.screens.scr2.@material": self.parse_material,  # e.g. 'scr_Transparent.par'
            "comp1.illumination.lmp1.@intensity": self.parse_other,  # e.g. 100
            "comp1.illumination.lmp1.@maxIglob": self.parse_other,  # e.g. 500
        }

    def parse(self):
        cp_vals = [self.parser_router[key](key) for key in CP_KEYS]
        return np.concatenate(cp_vals, axis=1)

    def parse_setpoint(self, key):
        v_night, v_day = self.action_dict[key]

        end_time = self.action_dict['comp1.illumination.lmp1.@endTime'].item()
        hours_light = self.action_dict['comp1.illumination.lmp1.@hoursLight'].item()

        t_rise, _ = ControlParser.get_sun_rise_and_set(self.date, self.city)
        t_start = min(t_rise, end_time - hours_light)
        t_end = end_time

        setpoints = [(t_start, v_night), (t_start + 1, v_day), (t_end - 1, v_day), (t_end, v_night)]

        t1, v1 = setpoints[0]
        t2, v2 = setpoints[-1]
        if t1 > 0:
            setpoints.insert(0, (0, v1))
        if t2 < 24:
            setpoints.append((24, v2))

        x = np.asarray([sp[0] for sp in setpoints])
        y = np.asarray([sp[1] for sp in setpoints])
        f = interp1d(x, y, axis=0)

        return f(np.arange(24)).reshape(24, 1)  # 24 x 1

    def parse_enable(self, key):
        enable = self.action_dict[key]
        val = [float(enable == flag) for flag in (True, False)]
        return np.repeat([val], 24, axis=0)  # 24 x 2

    def parse_material(self, key):
        material = self.action_dict[key]
        val = [float(material == x) for x in MATERIALS]
        return np.repeat([val], 24, axis=0)  # 24 x 3

    def parse_other(self, key):
        val = self.action_dict[key]
        val = np.asarray(val).reshape(1, 1)
        return np.repeat(val, 24, axis=0)  # 24 x 1


class ControlParser:
    def __init__(self, control, city_name=CITY_NAME):
        self.control = control
        self.start_date = datetime.date.fromisoformat(control['simset']['@startDate'])
        self.end_date = datetime.date.fromisoformat(control['simset']['@endDate'])
        self.city = lookup(city_name, database())
        self.num_days = (self.end_date - self.start_date).days
        self.parser_router = {
            "comp1.heatingpipes.pipe1.@maxTemp": self.parse_pipe_maxTemp,
            "comp1.heatingpipes.pipe1.@minTemp": self.parse_pipe_minTemp,
            "comp1.heatingpipes.pipe1.@radiationInfluence": self.parse_pipe_radInf,
            "comp1.setpoints.temp.@heatingTemp": self.parse_temp_heatingTemp,
            "comp1.setpoints.temp.@ventOffset": self.parse_temp_ventOffset,
            "comp1.setpoints.temp.@radiationInfluence": self.parse_temp_radInf,
            "comp1.setpoints.temp.@PbandVent": self.parse_temp_PbandVent,
            "comp1.setpoints.ventilation.@startWnd": self.parse_vent_startWnd,
            "comp1.setpoints.ventilation.@winLeeMin": self.parse_vent_winLeeMin,
            "comp1.setpoints.ventilation.@winLeeMax": self.parse_vent_winLeeMax,
            "comp1.setpoints.ventilation.@winWndMin": self.parse_vent_winWndMin,
            "comp1.setpoints.ventilation.@winWndMax": self.parse_vent_winWndMax,
            "common.CO2dosing.@pureCO2cap": self.parse_co2_pureCap,
            "comp1.setpoints.CO2.@setpoint": self.parse_co2_setpoint,
            "comp1.setpoints.CO2.@setpIfLamps": self.parse_co2_setpIfLamps,
            "comp1.setpoints.CO2.@doseCapacity": self.parse_co2_doseCap,
            "comp1.screens.scr1.@enabled": partial(self.parse_scr_enabled, scr_id=1),
            "comp1.screens.scr1.@material": partial(self.parse_scr_material, scr_id=1),
            "comp1.screens.scr1.@ToutMax": partial(self.parse_scr_ToutMax, scr_id=1),
            "comp1.screens.scr1.@closeBelow": partial(self.parse_scr_closeBelow, scr_id=1),
            "comp1.screens.scr1.@closeAbove": partial(self.parse_scr_closeAbove, scr_id=1),
            "comp1.screens.scr1.@lightPollutionPrevention": partial(self.parse_scr_LPP, scr_id=1),
            "comp1.screens.scr2.@enabled": partial(self.parse_scr_enabled, scr_id=2),
            "comp1.screens.scr2.@material": partial(self.parse_scr_material, scr_id=2),
            "comp1.screens.scr2.@ToutMax": partial(self.parse_scr_ToutMax, scr_id=2),
            "comp1.screens.scr2.@closeBelow": partial(self.parse_scr_closeBelow, scr_id=2),
            "comp1.screens.scr2.@closeAbove": partial(self.parse_scr_closeAbove, scr_id=2),
            "comp1.screens.scr2.@lightPollutionPrevention": partial(self.parse_scr_LPP, scr_id=2),
            "comp1.illumination.lmp1.@enabled": self.parse_lmp1_enabled,
            "comp1.illumination.lmp1.@intensity": self.parse_lmp1_intensity,
            "comp1.illumination.lmp1.@hoursLight": self.parse_lmp1_hoursLight,
            "comp1.illumination.lmp1.@endTime": self.parse_lmp1_endTime,
            "comp1.illumination.lmp1.@maxIglob": self.parse_lmp1_maxIglob,
            "comp1.illumination.lmp1.@maxPARsum": self.parse_lmp1_maxPARsum,
            "crp_lettuce.Intkam.management.@plantDensity": self.parse_plant_density,
        }

    def parse(self):
        assert self.end_date > self.start_date
        control = self.control

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
        return np.concatenate(cp_list, axis=1, dtype=np.float32)  # D x CP_DIM

    def parse2dict(self, keys):
        assert self.end_date > self.start_date
        # {key: value -> shape(D x 24 x num_dim_key)}
        return {key: self.parser_router[key](self.control) for key in keys}

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

    # @staticmethod
    # def schedule2arr(schedule, t_rise, t_set, preprocess):
    #     key_value_pairs = list(schedule.items())
    #     locals = {'r': t_rise, 's': t_set}
    #     convert = lambda t: eval(t, locals)
    #     kv_list = [[convert(t)] + preprocess(v) for t, v in key_value_pairs]
    #     return ControlParser.flatten(sorted(kv_list))

    @staticmethod
    def schedule2list(schedule, t_rise, t_set, preprocess):
        convert = lambda t: eval(t, {'r': t_rise, 's': t_set})
        setpoints = [(convert(t), preprocess(v)) for t, v in schedule.items()]
        setpoints = sorted(setpoints)

        t1, v1 = setpoints[0]
        t2, v2 = setpoints[-1]
        if t1 > 0:
            setpoints.insert(0, (0, v1))
        if t2 < 24:
            setpoints.append((24, v2))

        x = np.asarray([sp[0] for sp in setpoints])
        y = np.asarray([sp[1] for sp in setpoints])
        f = interp1d(x, y, axis=0)

        return f(np.arange(24)).tolist()  # 24 x N_dim

    def value2arr(self, value, preprocess, valid_dtype):
        if type(value) in valid_dtype:
            cp_list = [[preprocess(value)] * 24] * self.num_days
        elif type(value) == dict:
            cp_list = self.dict_value2list(value, valid_dtype, preprocess)
        return np.asarray(cp_list)  # D x 24 x N_dim

    def dict_value2list(self, value, valid_dtype, preprocess):
        offset_vals = []
        for date, day_value in value.items():
            day, month = date.split('-')
            day, month = int(day), int(month)
            dateinfo = datetime.date(2021, month, day)
            offset = max(0, (dateinfo - self.start_date).days)

            t_rise, t_set = self.get_sun_rise_and_set(dateinfo, self.city)
            if type(day_value) in valid_dtype:
                val = [preprocess(day_value)] * 24
            elif type(day_value) == dict:
                val = self.schedule2list(day_value, t_rise, t_set, preprocess)
            else:
                raise ValueError(f"invalid data type of {day_value}")

            offset_vals.append((offset, val))

        # fill values for missing days
        arr = [None] * self.num_days
        offset_last = 0
        for offset, val in offset_vals:
            for i in range(offset_last, offset + 1):
                arr[i] = val
            offset_last = offset

        for i in range(offset_last, self.num_days):
            arr[i] = val

        return arr  # D x 24 x N_dim

    def parse_pipe_maxTemp(self, control):
        value = self.get_value(control, "comp1.heatingpipes.pipe1.@maxTemp")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)

    def parse_pipe_minTemp(self, control):
        value = self.get_value(control, "comp1.heatingpipes.pipe1.@minTemp")
        preprocess = lambda x: [x]
        valid_dtype = [int, float]
        return self.value2arr(value, preprocess, valid_dtype)

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
        return self.value2arr(value, preprocess, valid_dtype)

    def parse_temp_ventOffset(self, control):
        value = self.get_value(control, "comp1.setpoints.temp.@ventOffset")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
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
        preprocess = lambda x: [x == True, x == False]
        return self.value2arr(value, preprocess, valid_dtype)

    def parse_scr_material(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@material")
        valid_dtype = [str]

        def preprocess(s):
            return [float(s == c) for c in MATERIALS]

        return self.value2arr(value, preprocess, valid_dtype)

    def parse_scr_ToutMax(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@ToutMax")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
        return self.value2arr(value, preprocess, valid_dtype)

    def parse_scr_closeBelow(self, control, scr_id):
        value = self.get_value(control, f"comp1.screens.scr{scr_id}.@closeBelow")
        valid_dtype = [int, float]
        preprocess = lambda x: [float(x)]
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
        arr = np.zeros((self.num_days, 24, 1))
        for day, density in setpoints:
            offset = day - 1
            if offset >= self.num_days:
                break
            arr[offset:, :, 0] = density
        return arr

    def parse_plant_density_to_setpoints(self, pd_str):
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

        self.check_plant_density(setpoints)
        return setpoints

    def check_plant_density(self, setpoints):
        days = [sp[0] for sp in setpoints]
        densities = [sp[1] for sp in setpoints]
        assert days[0] == 1  # must start with the 1st day
        assert sorted(days) == days  # days must be ascending
        assert sorted(densities, reverse=True) == densities  # densities must be descending


def agent_action_to_dict(action_arr: np.ndarray) -> OrderedDict:
    action_dict = OrderedDict()
    idx = 0
    for k, item in ACTION_PARAM_SPACE.items():
        # item is a 2-tuple
        size = len(item[0])
        action_dict[k] = action_arr[idx:idx + size]
        idx += size
    return action_dict


def agent_action_to_array(action_dict: OrderedDict) -> np.ndarray:
    action_arr = np.array([])
    for _, v in action_dict.items():
        action_arr = np.concatenate((action_arr, v), axis=None)
    return action_arr


def dump_actions(path: str, actions: np.ndarray):
    """
    Create a json file at PATH representing ACTIONS.
    Parameters
    ----------
    path: save file location.
    actions: shape (D, N_DIMS), the action history.
    """
    with open(BO_CONTROL_PATH) as f:
        bo_actions = json.load(f)
    actions = actions.astype(np.float)

    result = NestedDefaultDict()
    start_date = datetime.datetime.strptime(bo_actions['simset.@startDate'], '%Y-%m-%d').date()
    pd = bo_actions['init_plant_density']
    plant_densities = []

    # append RL actions to result
    for i, action in enumerate(actions):
        date = start_date + datetime.timedelta(days=i)
        action_dict = agent_action_to_dict(action)
        for k, v in action_dict:
            if k == 'end':
                continue
            elif k == 'crp_lettuce.Intkam.management.@plantDensity':
                pd_delta = v[0] if v[1] else 0
                if i == 0 or v[1]:
                    pd -= pd_delta
                    plant_densities.append(f'{i + 1} {pd}')
                continue
            elif k in ('comp1.setpoints.temp.@heatingTemp', 'comp1.setpoints.CO2.@setpoint'):
                city_info = lookup(CITY_NAME, database())
                t_rise, _ = ControlParser.get_sun_rise_and_set(date, city_info)
                end_time = action_dict['comp1.illumination.lmp1.@endTime']
                hours_light = action_dict['comp1.illumination.lmp1.@hoursLight']
                t_start = min(t_rise, end_time - hours_light)
                t_end = end_time
                v = {
                    t_start: v[0],
                    t_start + 1: v[1],
                    t_end - 1: v[1],
                    t_end: v[0]
                }

            # store into result
            result[k][date.strftime('%d-%m')] = v

    # parse plant density
    result['crp_lettuce.Intkam.management.@plantDensity'] = '; '.join(plant_densities)

    # add BO controls
    for k, v in bo_actions:
        if k != 'init_plant_density':
            result[k] = v
    sim_len = actions.shape[0]
    end_date = start_date + datetime.timedelta(days=sim_len)
    result['simset.@endDate'] = end_date.strftime('%Y-%m-%d')

    # add fixed controls
    for k, v in CONTROL_FIX:
        result[k] = v

    save_json_data(result, path)


# =========== Deprecated: don't use! ===============
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
            cp_arr = [preprocess(value)] * self.num_hours  # 1 x num_hours
        elif type(value) == dict:
            cp_arr = self.dict_value2arr(value, valid_dtype, preprocess, interpolate)
        return np.asarray(cp_arr).T  # N x num_hours

    def get_setpoints(value_dict):
        pass

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
                result[h_start:h_start + 24] = [preprocess(day_value)] * 24
            elif type(day_value) == dict:
                cp_arr_day = [None] * 24
                for hour, hour_value in day_value.items():
                    h_cur = round(eval(hour, {'r': r, 's': s}))
                    cp_arr_day[h_cur] = preprocess(hour_value)
                result[h_start:h_start + 24] = cp_arr_day
            else:
                raise ValueError

        # interpolate missing time hours
        points = [(i, v) for i, v in enumerate(result) if v is not None]
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        if x[0] > 0:
            x.insert(0, 0)
            y.insert(0, y[0])
        if x[-1] < self.num_hours - 1:
            x.append(self.num_hours - 1)
            y.append(y[-1])
        f = interp1d(x, y, kind=interpolate, axis=0)

        return f(range(self.num_hours))

    def parse_pipe_maxTemp(self, control):
        # sourcery skip: class-extract-method
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
            return [float(s == c) for c in choices]

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
            arr[0, hour_offset:] = density
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


def parse_control(control, keys):
    cp_dict = ControlParser(control).parse2dict(keys)
    # for key, arr in cp_dict.items():
    #     print(key, arr.shape)
    cp_list = [cp_dict[key] for key in keys]
    return np.concatenate(cp_list, axis=-1)  # D x 24 x CP_DIM


def parse_output(output, keys):
    output_vals = []
    for key in keys:
        val = output['data'][key]['data']
        val = [0 if x == 'NaN' else x for x in val]
        output_vals.append(val)
    output_vals = np.asarray(output_vals, dtype=np.float32)
    return output_vals.T  # T x NUM_KEYS


def parse_action(action_dict):
    return ActionParser(action_dict).parse()


def prepare_traces(data_dirs, save_dir, output_folder="outputs"):
    for data_dir in data_dirs:
        output_dir = os.path.join(data_dir, output_folder)
        print(f'preparing traces from {data_dir} ...')
        for name in tqdm(os.listdir(output_dir)):
            output = load_json_data(f'{output_dir}/{name}')
            if output['responsemsg'] != 'ok':
                continue

            ep = parse_output(output, EP_KEYS)  # T x EP_DIM
            op = parse_output(output, OP_KEYS)  # T x OP_DIM
            pl = parse_output(output, PL_KEYS)  # T x PL_DIM
            pd = parse_output(output, ["comp1.Plant.PlantDensity"])  # T x 1
            ph = parse_output(output, ["common.Economics.PeakHour"])  # T x 1

            ep_trace = ep.reshape(-1, 24, ep.shape[-1])  # D x 24 x EP_DIM
            op_trace = op.reshape(-1, 24, op.shape[-1])  # D x 24 x OP_DIM
            pl_trace = pl.reshape(-1, 24, pl.shape[-1])[:, 12]  # D x PL_DIM
            pd_trace = pd.reshape(-1, 24, pd.shape[-1])[:, 12]  # D x 1
            ph_trace = ph.reshape(-1, 24, ph.shape[-1])  # D x 24 x 1

            trace_dir = f"{save_dir}/{name[:-5]}"
            os.makedirs(trace_dir, exist_ok=True)

            np.save(f"{trace_dir}/ep_trace.npy", ep_trace)
            np.save(f"{trace_dir}/op_trace.npy", op_trace)
            np.save(f"{trace_dir}/pl_trace.npy", pl_trace)
            np.save(f"{trace_dir}/pd_trace.npy", pd_trace)
            np.save(f"{trace_dir}/ph_trace.npy", ph_trace)


def filter_jsons(filenames):
    return list(filter(lambda s: s.endswith('.json'), filenames))


def get_param_range(data_dirs, data_folder, parse_func, keys):
    arr_list = []
    print(f"collecting ranges for {keys}")
    for data_dir in data_dirs:
        print(f"collecting param ranges @ {data_dir}")
        names = os.listdir(f"{data_dir}/{data_folder}")
        for name in tqdm(filter_jsons(names)):
            path = f"{data_dir}/{data_folder}/{name}"
            arr = parse_func(load_json_data(path), keys)
            arr = arr.reshape(-1, arr.shape[-1])  # T x PARAM_DIM
            arr_list.append(arr)
    arr = np.concatenate(arr_list, axis=0)
    return arr.min(axis=0), arr.max(axis=0)


def get_min_max(arr):
    arr = arr.reshape(-1, arr.shape[-1])  # S x N_DIM
    return arr.min(0), arr.max(0)


# =========== Deprecated: don't use! ===============
class AGCDataset(Dataset):
    def __init__(self,
                 data_dirs,
                 processed_data_name='processed_data',
                 ep_keys=None,
                 op_keys=None,
                 force_preprocess=False,
                 ):
        super(AGCDataset, self).__init__()

        cp, ep, op, op_next = [], [], [], []
        for data_dir in data_dirs:
            data_path = f'{data_dir}/{processed_data_name}.npz'
            if not os.path.exists(data_path) or force_preprocess:
                self.preprocess_data(data_dir, processed_data_name, ep_keys, op_keys)
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

    def preprocess_data(self, data_dir, save_name, ep_keys, op_keys):
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

            control_vals = parse_control(control)  # control_vals: T x CP_DIM
            env_vals = parse_output(output, ep_keys)  # env_vals: T x EP_DIM
            output_vals = parse_output(output, op_keys)  # output_vals: T x OP_DIM

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

    def get_norm_data(self):
        data = self.get_data()

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


class ClimateDatasetHour(Dataset):
    EP_KEYS = [
        'common.Iglob.Value',
        'common.TOut.Value',
        'common.RHOut.Value',
        'common.Windsp.Value',
    ]
    OP_IN_KEYS = [
        # most important -> directly affect plant growth
        "comp1.Air.T",
        "comp1.Air.RH",
        "comp1.Air.ppm",
        "comp1.PARsensor.Above",
        # less important -> indirectly affect plant growth but directly affect costs
        "comp1.TPipe1.Value",
        "comp1.ConPipes.TSupPipe1",
        "comp1.PConPipe1.Value",
        "comp1.ConWin.WinLee",
        "comp1.ConWin.WinWnd",
        "comp1.Setpoints.SpHeat",
        "comp1.Setpoints.SpVent",
        "comp1.Scr1.Pos",
        "comp1.Scr2.Pos",
        "comp1.Lmp1.ElecUse",
        "comp1.McPureAir.Value",
    ]

    def __init__(self,
                 data_dirs,
                 norm_data,
                 ep_keys=None,
                 op_in_keys=None,
                 force_preprocess=False,
                 data_name="climate_model_data",
                 ) -> None:
        super().__init__()
        self.ep_keys = ep_keys or self.EP_KEYS
        self.op_in_keys = op_in_keys or self.OP_IN_KEYS
        self.data_name = data_name

        cp, ep, op_in, op_in_next = [], [], [], []
        for data_dir in data_dirs:
            data_path = f'{data_dir}/{data_name}.npz'
            if not os.path.exists(data_path) or force_preprocess:
                self.preprocess_data(data_dir)
            data = np.load(data_path)

            cp.append(data['cp'])
            ep.append(data['ep'])
            op_in.append(data['op_in'])
            op_in_next.append(data['op_in_next'])

        self.cp = np.concatenate(cp, axis=0, dtype=np.float32)
        self.ep = np.concatenate(ep, axis=0, dtype=np.float32)
        self.op_in = np.concatenate(op_in, axis=0, dtype=np.float32)
        self.op_in_next = np.concatenate(op_in_next, axis=0, dtype=np.float32)

        cp_mean, cp_std = norm_data['cp_mean'], norm_data['cp_std']
        ep_mean, ep_std = norm_data['ep_mean'], norm_data['ep_std']
        op_in_mean, op_in_std = norm_data['op_in_mean'], norm_data['op_in_std']

        self.cp_normed = normalize(self.cp, cp_mean, cp_std)
        self.ep_normed = normalize(self.ep, ep_mean, ep_std)
        self.op_in_normed = normalize(self.op_in, op_in_mean, op_in_std)
        self.op_in_next_normed = normalize(self.op_in_next, op_in_mean, op_in_std)

    def __getitem__(self, index):
        input = (self.cp_normed[index], self.ep_normed[index], self.op_in_normed[index])
        target = self.op_in_next_normed[index]
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
    def op_in_dim(self):
        return self.op_in.shape[1]

    @property
    def meta_data(self):
        return {
            'cp_dim': self.cp_dim,
            'ep_dim': self.ep_dim,
            'op_in_dim': self.op_in_dim,
        }

    def get_data(self):
        return {
            'cp': self.cp,
            'ep': self.ep,
            'op_in': self.op_in,
            'op_in_next': self.op_in_next
        }

    def preprocess_data(self, data_dir):
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

            control_vals = parse_control(control)  # control_vals: T x CP_DIM
            env_vals = parse_output(output, self.ep_keys)  # env_vals: T x EP_DIM
            output_vals = parse_output(output, self.op_in_keys)  # output_vals: T x OP_DIM

            # assumption: cp[t] + ep[t] + op[t] -> op[t+1]
            cp_vals = control_vals[:-1]
            ep_vals = env_vals[:-1]
            op_vals = output_vals[:-1]
            op_next_vals = output_vals[1:]

            assert cp_vals.shape[0] == ep_vals.shape[0] == op_vals.shape[0] == op_next_vals.shape[0]

            cp_arr.append(cp_vals)
            ep_arr.append(ep_vals)
            op_pre_arr.append(op_vals)
            op_cur_arr.append(op_next_vals)

        cp = np.concatenate(cp_arr, axis=0)
        ep = np.concatenate(ep_arr, axis=0)
        op_in = np.concatenate(op_pre_arr, axis=0)
        op_in_next = np.concatenate(op_cur_arr, axis=0)

        np.savez_compressed(
            f'{data_dir}/{self.data_name}.npz',
            cp=cp, ep=ep, op_in=op_in, op_in_next=op_in_next)

    @staticmethod
    def get_norm_data(norm_data_dirs):
        cp_list, ep_list, op_in_list = [], [], []
        for data_dir in norm_data_dirs:
            for name in os.listdir(f"{data_dir}/controls"):
                cp = load_json_data(f"{data_dir}/controls/{name}")
                cp_list.append(parse_control(cp))
            for name in os.listdir(f"{data_dir}/outputs"):
                output = load_json_data(f"{data_dir}/outputs/{name}")
                ep = parse_output(output, ClimateDatasetHour.EP_KEYS)
                op_in = parse_output(output, ClimateDatasetHour.OP_IN_KEYS)
                ep_list.append(ep)
                op_in_list.append(op_in)

        cp = np.concatenate(cp_list, axis=0)
        ep = np.concatenate(ep_list, axis=0)
        op_in = np.concatenate(op_in_list, axis=0)

        return {
            'cp_mean': np.mean(cp, axis=0), 'cp_std': np.std(cp, axis=0),
            'ep_mean': np.mean(ep, axis=0), 'ep_std': np.std(ep, axis=0),
            'op_in_mean': np.mean(op_in, axis=0), 'op_in_std': np.std(op_in, axis=0),
        }


class PlantDatasetHour(Dataset):
    OP_IN_KEYS = [
        "comp1.Air.T",
        "comp1.Air.RH",
        "comp1.Air.ppm",
        "comp1.PARsensor.Above",
        "comp1.Plant.PlantDensity",
    ]
    OP_PL_KEYS = [
        "comp1.Plant.headFW",
        "comp1.Plant.shootDryMatterContent",
        "comp1.Plant.qualityLoss"
    ]

    def __init__(self,
                 data_dirs,
                 norm_data,
                 op_in_keys=None,
                 op_pl_keys=None,
                 force_preprocess=False,
                 data_name="plant_model_data",
                 ) -> None:
        super().__init__()
        self.op_in_keys = op_in_keys or self.OP_IN_KEYS
        self.op_pl_keys = op_pl_keys or self.OP_PL_KEYS
        self.data_name = data_name

        op_in, pl, pl_next = [], [], []
        for data_dir in data_dirs:
            data_path = f'{data_dir}/{data_name}.npz'
            if not os.path.exists(data_path) or force_preprocess:
                self.preprocess_data(data_dir)
            data = np.load(data_path)

            op_in.append(data['op_in'])
            pl.append(data['pl'])
            pl_next.append(data['pl_next'])

        self.op_in = np.concatenate(op_in, axis=0, dtype=np.float32)
        self.pl = np.concatenate(pl, axis=0, dtype=np.float32)
        self.pl_next = np.concatenate(pl_next, axis=0, dtype=np.float32)

        op_in_mean, op_in_std = norm_data['op_in_mean'], norm_data['op_in_std']
        op_pl_mean, op_pl_std = norm_data['op_pl_mean'], norm_data['op_pl_std']

        self.op_in_normed = normalize(self.op_in, op_in_mean, op_in_std)
        self.pl_normed = normalize(self.pl, op_pl_mean, op_pl_std)
        self.pl_next_normed = normalize(self.pl_next, op_pl_mean, op_pl_std)

    def __getitem__(self, index):
        input = (self.op_in_normed[index], self.pl_normed[index])
        target = self.pl_next_normed[index]
        return input, target

    def __len__(self):
        return self.op_in.shape[0]

    @property
    def op_in_dim(self):
        return self.op_in.shape[1]

    @property
    def op_pl_dim(self):
        return self.pl.shape[1]

    @property
    def meta_data(self):
        return {
            'op_in_dim': self.op_in_dim,
            'op_pl_dim': self.op_pl_dim,
        }

    def get_data(self):
        return {
            'op_in': self.op_in,
            'pl': self.pl,
            'pl_next': self.pl_next
        }

    def preprocess_data(self, data_dir):
        control_dir = os.path.join(data_dir, 'controls')
        output_dir = os.path.join(data_dir, 'outputs')

        assert os.path.exists(control_dir)
        assert os.path.exists(output_dir)

        cp_names = os.listdir(control_dir)
        op_names = os.listdir(output_dir)
        names = list(set(cp_names).intersection(set(op_names)))

        print(f'preprocessing data @ {data_dir} ...')

        op_in_arr, op_pl_arr, op_pl_next_arr = [], [], []
        for name in tqdm(names):
            output = load_json_data(os.path.join(output_dir, name))

            if output['responsemsg'] != 'ok':
                continue

            in_climate_vals = parse_output(output, self.op_in_keys)  # env_vals: T x EP_DIM
            plant_vals = parse_output(output, self.op_pl_keys)  # output_vals: T x OP_DIM

            # assumption: op_in[t] + pl[t] -> pl[t+1]
            op_in_vals = in_climate_vals[:-1]
            op_pl_vals = plant_vals[:-1]
            op_pl_next_vals = plant_vals[1:]

            assert op_in_vals.shape[0] == op_pl_vals.shape[0] == op_pl_next_vals.shape[0]

            op_in_arr.append(op_in_vals)
            op_pl_arr.append(op_pl_vals)
            op_pl_next_arr.append(op_pl_next_vals)

        op_in_arr = np.concatenate(op_in_arr, axis=0)
        op_pl_arr = np.concatenate(op_pl_arr, axis=0)
        op_pl_next_arr = np.concatenate(op_pl_next_arr, axis=0)

        np.savez_compressed(
            f'{data_dir}/{self.data_name}.npz',
            op_in=op_in_arr, pl=op_pl_arr, pl_next=op_pl_next_arr)

    @staticmethod
    def get_norm_data(norm_data_dirs):
        op_in_list, op_pl_list = [], []
        for data_dir in norm_data_dirs:
            for name in os.listdir(f"{data_dir}/outputs"):
                output = load_json_data(f"{data_dir}/outputs/{name}")
                op_in = parse_output(output, PlantDatasetHour.OP_IN_KEYS)
                pl = parse_output(output, PlantDatasetHour.OP_PL_KEYS)
                op_in_list.append(op_in)
                op_pl_list.append(pl)

        op_in = np.concatenate(op_in_list, axis=0)
        pl = np.concatenate(op_pl_list, axis=0)

        return {
            'op_in_mean': np.mean(op_in, axis=0), 'op_in_std': np.std(op_in, axis=0),
            'op_pl_mean': np.mean(pl, axis=0), 'op_pl_std': np.std(pl, axis=0),
        }


class ClimateDatasetDay(Dataset):
    def __init__(self,
                 data_dirs,
                 norm_data=None,
                 cp_keys=CP_KEYS,
                 ep_keys=EP_KEYS,
                 op_keys=OP_KEYS,
                 force_preprocess=False,
                 data_name="climate_data_day",
                 control_folder="controls",
                 output_folder="outputs",
                 ) -> None:
        super().__init__()
        self.cp_keys = cp_keys
        self.ep_keys = ep_keys
        self.op_keys = op_keys
        self.data_name = data_name
        self.control_folder = control_folder
        self.output_folder = output_folder
        self.norm_data = norm_data
        self.preprocess(data_dirs, force_preprocess)

        if not self.norm_data:
            # self.norm_data = {
            #     'cp': get_param_range(data_dirs, control_folder, parse_control, cp_keys),
            #     'ep': get_param_range(data_dirs, output_folder, parse_output, ep_keys),
            #     'op': get_param_range(data_dirs, output_folder, parse_output, op_keys),
            # }
            self.norm_data = {
                'cp': get_min_max(self.cp),
                'ep': get_min_max(self.ep),
                'op': get_min_max(np.concatenate([self.op, self.op_next], axis=0)),
            }

        self.cp_normed = normalize_zero2one(self.cp, self.norm_data['cp'])  # D x 24 x CP_DIM
        self.ep_normed = normalize_zero2one(self.ep, self.norm_data['ep'])  # D x 24 x EP_DIM
        self.op_normed = normalize_zero2one(self.op, self.norm_data['op'])  # D x 24 x OP_DIM
        self.op_next_normed = normalize_zero2one(self.op_next, self.norm_data['op'])  # D x 24 x OP_DIM

        self.cp_normed = self.cp_normed.astype(np.float32)
        self.ep_normed = self.ep_normed.astype(np.float32)
        self.op_normed = self.op_normed.astype(np.float32)
        self.op_next_normed = self.op_next_normed.astype(np.float32)

    def __getitem__(self, index):
        cp = self.cp_normed[index].flatten()  # n_dim = 24 x CP_DIM
        ep = self.ep_normed[index].flatten()  # n_dim = 24 x EP_DIM
        op = self.op_normed[index].flatten()  # n_dim = 24 x OP_DIM
        op_next = self.op_next_normed[index].flatten()  # n_dim = 24 x OP_DIM
        return (cp, ep, op), op_next

    def __len__(self):
        return self.cp_normed.shape[0]

    def preprocess(self, data_dirs, force_preprocess):
        cp, ep, op, op_next = [], [], [], []
        for data_dir in data_dirs:
            data_path = f'{data_dir}/{self.data_name}.npz'
            if not os.path.exists(data_path) or force_preprocess:
                self.preprocess_dir(data_dir)

            data = np.load(data_path)
            cp.append(data['cp'])  # D_i x 24 x CP_DIM
            ep.append(data['ep'])  # D_i x 24 x EP_DIM
            op.append(data['op'])  # D_i x 24 x OP_DIM
            op_next.append(data['op_next'])  # D_i x 24 x OP_DIM

        self.cp = np.concatenate(cp, axis=0)  # SUM(D_i) x 24 x CP_DIM
        self.ep = np.concatenate(ep, axis=0)  # SUM(D_i) x 24 x EP_DIM
        self.op = np.concatenate(op, axis=0)  # SUM(D_i) x 24 x OP_DIM
        self.op_next = np.concatenate(op_next, axis=0)  # SUM(D_i) x 24 x OP_DIM

    def preprocess_dir(self, data_dir):
        control_dir = os.path.join(data_dir, self.control_folder)
        output_dir = os.path.join(data_dir, self.output_folder)

        assert os.path.exists(control_dir)
        assert os.path.exists(output_dir)

        cp_names = os.listdir(control_dir)
        op_names = os.listdir(output_dir)
        names = list(set(cp_names).intersection(set(op_names)))

        print(f'preprocessing data @ {data_dir} ...')

        cp_arr, ep_arr, op_arr, op_next_arr = [], [], [], []
        for name in tqdm(filter_jsons(names)):
            control = load_json_data(os.path.join(control_dir, name))
            output = load_json_data(os.path.join(output_dir, name))

            if output['responsemsg'] != 'ok':
                continue

            # ========= Parse CP params =========
            cp = parse_control(control, self.cp_keys)  # D x 24 x CP_DIM

            # ========= Parse EP params =========
            ep = parse_output(output, self.ep_keys)  # T x EP_DIM
            ep = ep.reshape(-1, 24, ep.shape[-1])  # D x 24 x EP_DIM

            # ========= Parse OP params ==========
            op = parse_output(output, self.op_keys)  # T x OP_DIM
            # op_init = np.repeat(op[:1], 24, axis=0) # 24 x OP_DIM
            op_init = np.zeros((24, op.shape[-1]))  # 24 x OP_DIM
            op = np.concatenate([op_init, op], axis=0)  # (T+24) x OP_DIM
            op = op.reshape(-1, 24, op.shape[-1])  # (D+1) x 24 x OP_DIM

            # assumption: cp[d+1] + ep[d+1] + op[d] -> op[d+1]
            op_next = op[1:]  # D x 24 x OP_DIM
            op = op[:-1]  # D x 24 x OP_DIM

            cp_arr.append(cp)
            ep_arr.append(ep)
            op_arr.append(op)
            op_next_arr.append(op_next)

        cp = np.concatenate(cp_arr, axis=0)
        ep = np.concatenate(ep_arr, axis=0)
        op = np.concatenate(op_arr, axis=0)
        op_next = np.concatenate(op_next_arr, axis=0)

        np.savez_compressed(
            f'{data_dir}/{self.data_name}.npz',
            cp=cp, ep=ep, op=op, op_next=op_next)

    def get_meta_data(self):
        return {
            'cp_dim': self.cp.shape[-1] * 24,
            'ep_dim': self.ep.shape[-1] * 24,
            'op_dim': self.op.shape[-1] * 24,
            'norm_data': self.norm_data,
        }


class PlantDatasetDay(Dataset):
    def __init__(self,
                 data_dirs,
                 norm_data=None,
                 op_in_keys=OP_IN_KEYS,
                 pl_keys=PL_KEYS,
                 force_preprocess=False,
                 data_name="plant_data_day",
                 control_folder="controls",
                 output_folder="outputs",
                 ) -> None:
        super().__init__()
        self.op_in_keys = op_in_keys
        self.pl_keys = pl_keys
        self.pd_keys = ["crp_lettuce.Intkam.management.@plantDensity"]
        self.data_name = data_name
        self.control_folder = control_folder
        self.output_folder = output_folder
        self.pl_init = np.asarray([PL_INIT_VALUE[key] for key in self.pl_keys])
        self.norm_data = norm_data
        self.preprocess(data_dirs, force_preprocess)

        if not self.norm_data:
            # self.norm_data = {
            #     'pd': get_param_range(data_dirs, control_folder, parse_control, self.pd_keys),
            #     'op_in': get_param_range(data_dirs, output_folder, parse_output, op_in_keys),
            #     'pl': get_param_range(data_dirs, output_folder, parse_output, pl_keys),
            # }
            self.norm_data = {
                'pd': get_min_max(self.pd),
                'op_in': get_min_max(self.op_in),
                'pl': get_min_max(np.concatenate([self.pl, self.pl_next], axis=0)),
            }

        self.pd_normed = normalize_zero2one(self.pd, self.norm_data['pd'])  # D x 24 x OP_IN_DIM
        self.op_in_normed = normalize_zero2one(self.op_in, self.norm_data['op_in'])  # D x 24 x OP_IN_DIM
        self.pl_normed = normalize_zero2one(self.pl, self.norm_data['pl'])  # D x 24 x PL_DIM
        self.pl_next_normed = normalize_zero2one(self.pl_next, self.norm_data['pl'])  # D x 24 x PL_DIM

        self.pd_normed = self.pd_normed.astype(np.float32)
        self.op_in_normed = self.op_in_normed.astype(np.float32)
        self.pl_normed = self.pl_normed.astype(np.float32)
        self.pl_next_normed = self.pl_next_normed.astype(np.float32)

    def __getitem__(self, index):
        op_in = self.op_in_normed[index].flatten()  # n_dim = 24 x OP_IN_DIM
        pd = self.pd_normed[index, 12].flatten()  # n_dim = 1
        pl = self.pl_normed[index, 12].flatten()  # n_dim = PL_DIM
        pl_next = self.pl_next_normed[index, 12].flatten()  # n_dim = PL_DIM
        return (pd, op_in, pl), pl_next

    def __len__(self):
        return self.op_in_normed.shape[0]

    def preprocess(self, data_dirs, force_preprocess):
        pd, op_in, pl, pl_next = [], [], [], []
        for data_dir in data_dirs:
            data_path = f'{data_dir}/{self.data_name}.npz'
            if not os.path.exists(data_path) or force_preprocess:
                self.preprocess_dir(data_dir)
            data = np.load(data_path)

            pd.append(data['pd'])
            op_in.append(data['op_in'])
            pl.append(data['pl'])
            pl_next.append(data['pl_next'])

        self.pd = np.concatenate(pd, axis=0)  # D x 24 x 1
        self.op_in = np.concatenate(op_in, axis=0)  # D x 24 x OP_IN_DIM
        self.pl = np.concatenate(pl, axis=0)  # D x 24 x PL_DIM
        self.pl_next = np.concatenate(pl_next, axis=0)  # D x 24 x PL_DIM

    def preprocess_dir(self, data_dir):
        control_dir = os.path.join(data_dir, self.control_folder)
        output_dir = os.path.join(data_dir, self.output_folder)

        assert os.path.exists(control_dir)
        assert os.path.exists(output_dir)

        cp_names = os.listdir(control_dir)
        op_names = os.listdir(output_dir)
        names = list(set(cp_names).intersection(set(op_names)))

        print(f'preprocessing data @ {data_dir} ...')

        pd_arr, op_in_arr, pl_arr, pl_next_arr = [], [], [], []
        for name in tqdm(filter_jsons(names)):
            output = load_json_data(os.path.join(output_dir, name))
            control = load_json_data(os.path.join(control_dir, name))

            if output['responsemsg'] != 'ok':
                continue

            # ============ Parse PlantDensity ===============
            pd = parse_control(control, self.pd_keys)  # D x 24 x 1

            # ============ Parse Inside Params ==============
            op_in = parse_output(output, self.op_in_keys)  # T x OP_IN_DIM
            op_in = op_in.reshape(-1, 24, op_in.shape[-1])  # D x 24 x OP_IN_DIM

            # ============ Parse Plant Params ==============
            pl = parse_output(output, self.pl_keys)  # T x PL_DIM
            pl_init = np.repeat(self.pl_init.reshape(1, pl.shape[-1]), 24, axis=0)  # 24 x PL_DIM
            pl = np.concatenate([pl_init, pl], axis=0)  # (T+24) x PL_DIM
            pl = pl.reshape(-1, 24, pl.shape[-1])  # (D+1) x 24 x PL_DIM

            # assumption: op_in[d+1] + pl[d] -> pl[d+1]
            pl_next = pl[1:]  # D x 24 x PL_DIM
            pl = pl[:-1]  # D x 24 x PL_DIM

            pd_arr.append(pd)
            op_in_arr.append(op_in)
            pl_arr.append(pl)
            pl_next_arr.append(pl_next)

        pd_arr = np.concatenate(pd_arr, axis=0)
        op_in_arr = np.concatenate(op_in_arr, axis=0)
        pl_arr = np.concatenate(pl_arr, axis=0)
        pl_next_arr = np.concatenate(pl_next_arr, axis=0)

        np.savez_compressed(
            f'{data_dir}/{self.data_name}.npz',
            pd=pd_arr, op_in=op_in_arr, pl=pl_arr, pl_next=pl_next_arr)

    def get_meta_data(self):
        return {
            'op_in_dim': self.op_in.shape[-1] * 24,
            'pl_dim': self.pl.shape[-1],
            'norm_data': self.norm_data,
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dirs', type=str, nargs="+", required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--output-folder', type=str, default="outputs")
    args = parser.parse_args()

    prepare_traces(args.data_dirs, args.save_dir, args.output_folder)
