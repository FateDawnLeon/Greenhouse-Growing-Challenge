import os
import datetime
import numpy as np

from torch.utils.data import Dataset

from tqdm import tqdm
from astral.sun import sun
from astral.geocoder import lookup, database
from scipy.interpolate import interp1d

from constant import START_DATE, CITY_NAME, ENV_KEYS, OUTPUT_KEYS
from utils import load_json_data, normalize, normalize_zero2one


class ControlParser:
    def __init__(self, control, city_name='Amsterdam'):
        self.control = control
        self.start_date = datetime.date.fromisoformat(control['simset']['@startDate'])
        self.end_date = datetime.date.fromisoformat(control['simset']['@endDate'])
        self.city = lookup(city_name, database())
        self.num_days = (self.end_date - self.start_date).days

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

    def parse2dict(self):
        assert self.end_date > self.start_date
        control = self.control
        
        return {
            "comp1.heatingpipes.pipe1.@maxTemp": self.parse_pipe_maxTemp(control),
            "comp1.heatingpipes.pipe1.@minTemp": self.parse_pipe_minTemp(control),
            "comp1.heatingpipes.pipe1.@radiationInfluence": self.parse_pipe_radInf(control),
            "comp1.setpoints.temp.@heatingTemp": self.parse_temp_heatingTemp(control),
            "comp1.setpoints.temp.@ventOffset": self.parse_temp_ventOffset(control),
            "comp1.setpoints.temp.@radiationInfluence": self.parse_temp_radInf(control),
            "comp1.setpoints.temp.@PbandVent": self.parse_temp_PbandVent(control),
            "comp1.setpoints.ventilation.@startWnd": self.parse_vent_startWnd(control),
            "comp1.setpoints.ventilation.@winLeeMin": self.parse_vent_winLeeMin(control),
            "comp1.setpoints.ventilation.@winLeeMax": self.parse_vent_winLeeMax(control),
            "comp1.setpoints.ventilation.@winWndMin": self.parse_vent_winWndMin(control),
            "comp1.setpoints.ventilation.@winWndMax": self.parse_vent_winWndMax(control),
            "common.CO2dosing.@pureCO2cap": self.parse_co2_pureCap(control),
            "comp1.setpoints.CO2.@setpoint": self.parse_co2_setpoint(control),
            "comp1.setpoints.CO2.@setpIfLamps": self.parse_co2_setpIfLamps(control),
            "comp1.setpoints.CO2.@doseCapacity": self.parse_co2_doseCap(control),
            "comp1.screens.scr1.@enabled": self.parse_scr_enabled(control, 1),
            "comp1.screens.scr1.@material": self.parse_scr_material(control, 1),
            "comp1.screens.scr1.@ToutMax": self.parse_scr_ToutMax(control, 1),
            "comp1.screens.scr1.@closeBelow": self.parse_scr_closeBelow(control, 1),
            "comp1.screens.scr1.@closeAbove": self.parse_scr_closeAbove(control, 1),
            "comp1.screens.scr1.@lightPollutionPrevention": self.parse_scr_LPP(control, 1),
            "comp1.screens.scr2.@enabled": self.parse_scr_enabled(control, 2),
            "comp1.screens.scr2.@material": self.parse_scr_material(control, 2),
            "comp1.screens.scr2.@ToutMax": self.parse_scr_ToutMax(control, 2),
            "comp1.screens.scr2.@closeBelow": self.parse_scr_closeBelow(control, 2),
            "comp1.screens.scr2.@closeAbove": self.parse_scr_closeAbove(control, 2),
            "comp1.screens.scr2.@lightPollutionPrevention": self.parse_scr_LPP(control, 2),
            "comp1.illumination.lmp1.@enabled": self.parse_lmp1_enabled(control),
            "comp1.illumination.lmp1.@intensity": self.parse_lmp1_intensity(control),
            "comp1.illumination.lmp1.@hoursLight": self.parse_lmp1_hoursLight(control),
            "comp1.illumination.lmp1.@endTime": self.parse_lmp1_endTime(control),
            "comp1.illumination.lmp1.@maxIglob": self.parse_lmp1_maxIglob(control),
            "comp1.illumination.lmp1.@maxPARsum": self.parse_lmp1_maxPARsum(control),
            "crp_lettuce.Intkam.management.@plantDensity": self.parse_plant_density(control),
        }  # {key: value(n_dim x D)}

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

    @staticmethod
    def schedule2arr(schedule, t_rise, t_set, preprocess):
        key_value_pairs = list(schedule.items())
        val_map = {'r': t_rise, 's': t_set}
        convert = lambda t: float(t) if t.isdigit() else val_map[t]
        kv_list = [[convert(t)] + preprocess(v) for t, v in key_value_pairs]
        return ControlParser.flatten(sorted(kv_list))

    def value2arr(self, value, preprocess, valid_dtype):
        if type(value) in valid_dtype:
            cp_arr = [preprocess(value)] * self.num_days
        elif type(value) == dict:
            cp_arr = self.dict_value2arr(value, valid_dtype, preprocess)
        return np.asarray(cp_arr, dtype=np.float32)  # D x N_dim

    def dict_value2arr(self, value, valid_dtype, preprocess):
        offset_vals = []
        for date, day_value in value.items():
            day, month = date.split('-')
            day, month = int(day), int(month)
            dateinfo = datetime.date(2021, month, day)
            offset = max(0, (dateinfo - self.start_date).days)
            
            t_rise, t_set = self.get_sun_rise_and_set(dateinfo, self.city)
            if type(day_value) in valid_dtype:
                val = preprocess(day_value)
            elif type(day_value) == dict:
                val = self.schedule2arr(day_value, t_rise, t_set, preprocess)
            else:
                raise ValueError(f"invalid data type of {day_value}")

            offset_vals.append((offset, val))

        # fill values for missing days
        arr = [None] * self.num_days
        offset_last = 0
        for offset, val in offset_vals:
            for i in range(offset_last, offset+1):
                arr[i] = val
            offset_last = offset
        
        for i in range(offset_last, self.num_days):
            arr[i] = val

        return arr  # D x N_dim
    
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
        arr = np.zeros((self.num_days, 1))
        for day, density in setpoints:
            offset = day - 1
            if offset >= self.num_days:
                break
            arr[offset:, 0] = density
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


def parse_output(output, keys):
    output_vals = []
    for key in keys:
        val = output['data'][key]['data']
        val = [0 if x == 'NaN' else x for x in val]
        output_vals.append(val)
    output_vals = np.asarray(output_vals, dtype=np.float32)
    return output_vals.T # T x NUM_KEYS


def prepare_traces(data_dirs, save_dir):
    for data_dir in data_dirs:
        output_dir = os.path.join(data_dir, 'outputs')
        print(f'preparing traces from {data_dir} ...')
        for name in tqdm(os.listdir(output_dir)):
            output = load_json_data(f'{output_dir}/{name}')
            if output['responsemsg'] != 'ok':
                continue
            
            ep = parse_output(output, ClimateDatasetDay.EP_KEYS)  # T x EP_DIM
            op = parse_output(output, ClimateDatasetDay.OP_KEYS)  # T x OP_DIM
            pl = parse_output(output, PlantDatasetDay.OP_PL_KEYS)  # T x OP_PL_DIM

            ep_trace = ep.reshape(-1, 24, ep.shape[-1])  # D x 24 x EP_DIM
            op_trace = op.reshape(-1, 24, op.shape[-1])  # D x 24 x OP_DIM
            pl_trace = pl.reshape(-1, 24, pl.shape[-1]).mean(axis=1)  # D x OP_PL_DIM

            trace_dir = f"{save_dir}/{name[:-5]}"
            os.makedirs(trace_dir, exist_ok=True)

            np.save(f"{trace_dir}/ep_trace.npy", ep_trace)
            np.save(f"{trace_dir}/op_trace.npy", op_trace)
            np.save(f"{trace_dir}/pl_trace.npy", pl_trace)


def compute_mean_std(data_dirs, is_control, keys=None):
    data_list = []
    
    for data_dir in data_dirs:
        if is_control:
            for name in os.listdir(f"{data_dir}/controls"):
                control = load_json_data(f"{data_dir}/controls/{name}")
                data_list.append(parse_control(control))
        else:
            for name in os.listdir(f"{data_dir}/outputs"):
                output = load_json_data(f"{data_dir}/outputs/{name}")
                data_list.append(parse_output(output, keys)) 

    data = np.concatenate(data_list, axis=0, dtype=np.float32)
    return np.mean(data, axis=0), np.std(data, axis=0)


def get_control_range(data_dirs):
    cp_list = []
    for data_dir in data_dirs:
        print(f"collecting control param ranges @ {data_dir}")
        names = os.listdir(f"{data_dir}/controls")
        for name in tqdm(names):
            control_path = f"{data_dir}/controls/{name}"
            control = load_json_data(control_path)
            cp = parse_control(control)
            cp_list.append(cp)
    cp = np.concatenate(cp_list, axis=0)
    return cp.min(axis=0), cp.max(axis=0)


def get_output_range(data_dirs, keys):
    op_list = []
    for data_dir in data_dirs:
        print(f"collecting output param ranges @ {data_dir}")
        names = os.listdir(f"{data_dir}/controls")
        for name in tqdm(names):
            output_path = f"{data_dir}/outputs/{name}"
            output = load_json_data(output_path)
            op = parse_output(output, keys)
            op_list.append(op)
    op = np.concatenate(op_list, axis=0)
    return op.min(axis=0), op.max(axis=0)


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

            control_vals = parse_control(control) # control_vals: T x CP_DIM
            env_vals = parse_output(output, ep_keys) # env_vals: T x EP_DIM
            output_vals = parse_output(output, op_keys) # output_vals: T x OP_DIM

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

            control_vals = parse_control(control) # control_vals: T x CP_DIM
            env_vals = parse_output(output, self.ep_keys) # env_vals: T x EP_DIM
            output_vals = parse_output(output, self.op_in_keys) # output_vals: T x OP_DIM

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
                ep = parse_output(output, GreenhouseClimateDataset.EP_KEYS)
                op_in = parse_output(output, GreenhouseClimateDataset.OP_IN_KEYS)
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

        op_in, op_pl, op_pl_next = [], [], []
        for data_dir in data_dirs:
            data_path = f'{data_dir}/{data_name}.npz'
            if not os.path.exists(data_path) or force_preprocess:
                self.preprocess_data(data_dir)
            data = np.load(data_path)

            op_in.append(data['op_in'])
            op_pl.append(data['op_pl'])
            op_pl_next.append(data['op_pl_next'])

        self.op_in = np.concatenate(op_in, axis=0, dtype=np.float32)
        self.op_pl = np.concatenate(op_pl, axis=0, dtype=np.float32)
        self.op_pl_next = np.concatenate(op_pl_next, axis=0, dtype=np.float32)

        op_in_mean, op_in_std = norm_data['op_in_mean'], norm_data['op_in_std']
        op_pl_mean, op_pl_std = norm_data['op_pl_mean'], norm_data['op_pl_std']
        
        self.op_in_normed = normalize(self.op_in, op_in_mean, op_in_std)
        self.op_pl_normed = normalize(self.op_pl, op_pl_mean, op_pl_std)
        self.op_pl_next_normed = normalize(self.op_pl_next, op_pl_mean, op_pl_std)

    def __getitem__(self, index):
        input = (self.op_in_normed[index], self.op_pl_normed[index])
        target = self.op_pl_next_normed[index] 
        return input, target

    def __len__(self):
        return self.op_in.shape[0]

    @property
    def op_in_dim(self):
        return self.op_in.shape[1]
    
    @property
    def op_pl_dim(self):
        return self.op_pl.shape[1]

    @property
    def meta_data(self):
        return {
            'op_in_dim': self.op_in_dim,
            'op_pl_dim': self.op_pl_dim,
        }

    def get_data(self):
        return {
            'op_in': self.op_in, 
            'op_pl': self.op_pl, 
            'op_pl_next': self.op_pl_next
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

            in_climate_vals = parse_output(output, self.op_in_keys) # env_vals: T x EP_DIM
            plant_vals = parse_output(output, self.op_pl_keys) # output_vals: T x OP_DIM

            # assumption: op_in[t] + op_pl[t] -> op_pl[t+1]
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
            op_in=op_in_arr, op_pl=op_pl_arr, op_pl_next=op_pl_next_arr)

    @staticmethod
    def get_norm_data(norm_data_dirs):
        op_in_list, op_pl_list = [], []
        for data_dir in norm_data_dirs:
            for name in os.listdir(f"{data_dir}/outputs"):
                output = load_json_data(f"{data_dir}/outputs/{name}")
                op_in = parse_output(output, PlantDatasetHour.OP_IN_KEYS)
                op_pl = parse_output(output, PlantDatasetHour.OP_PL_KEYS)
                op_in_list.append(op_in)
                op_pl_list.append(op_pl)
        
        op_in = np.concatenate(op_in_list, axis=0)
        op_pl = np.concatenate(op_pl_list, axis=0)
        
        return {
            'op_in_mean': np.mean(op_in, axis=0), 'op_in_std': np.std(op_in, axis=0),
            'op_pl_mean': np.mean(op_pl, axis=0), 'op_pl_std': np.std(op_pl, axis=0),
        }


class ClimateDatasetDay(Dataset):
    EP_KEYS = [
        'common.Iglob.Value',
        'common.TOut.Value',
        'common.RHOut.Value',
        'common.Windsp.Value',
    ]
    OP_KEYS = [
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
        ranges, 
        force_preprocess=False, 
        data_name="climate_data_day"
    ) -> None:
        super().__init__()
        self.data_name = data_name
        self.ranges = ranges

        cp, ep, op, op_next = [], [], [], []
        for data_dir in data_dirs:
            data_path = f'{data_dir}/{data_name}.npz'
            if not os.path.exists(data_path) or force_preprocess:
                self.preprocess_data(data_dir)
            data = np.load(data_path)

            cp.append(data['cp'])  # D_i x 24 x CP_DIM
            ep.append(data['ep'])  # D_i x 24 x EP_DIM
            op.append(data['op'])  # D_i x 24 x OP_DIM
            op_next.append(data['op_next'])  # D_i x 24 x OP_DIM

        self.cp = np.concatenate(cp, axis=0, dtype=np.float32)  # SUM(D_i) x 24 x CP_DIM
        self.ep = np.concatenate(ep, axis=0, dtype=np.float32)  # SUM(D_i) x 24 x EP_DIM
        self.op = np.concatenate(op, axis=0, dtype=np.float32)  # SUM(D_i) x 24 x OP_DIM
        self.op_next = np.concatenate(op_next, axis=0, dtype=np.float32)  # SUM(D_i) x 24 x OP_DIM

        self.cp_dim = self.cp.shape[-1] * 24
        self.ep_dim = self.ep.shape[-1] * 24
        self.op_dim = self.op.shape[-1] * 24

        self.cp_normed = normalize_zero2one(self.cp, ranges['cp']).reshape(-1, self.cp_dim)
        self.ep_normed = normalize_zero2one(self.ep, ranges['ep']).reshape(-1, self.ep_dim)
        self.op_normed = normalize_zero2one(self.op, ranges['op']).reshape(-1, self.op_dim)
        self.op_next_normed = normalize_zero2one(self.op_next, ranges['op']).reshape(-1, self.op_dim)

    def __getitem__(self, index):
        input = (self.cp_normed[index], self.ep_normed[index], self.op_normed[index])
        target = self.op_next_normed[index]
        return input, target

    def __len__(self):
        return self.cp_normed.shape[0]

    def preprocess_data(self, data_dir):
        control_dir = os.path.join(data_dir, 'controls')
        output_dir = os.path.join(data_dir, 'outputs')

        assert os.path.exists(control_dir)
        assert os.path.exists(output_dir)

        cp_names = os.listdir(control_dir)
        op_names = os.listdir(output_dir)
        names = list(set(cp_names).intersection(set(op_names)))

        print(f'preprocessing data @ {data_dir} ...')

        cp_arr, ep_arr, op_arr, op_next_arr = [], [], [], []
        for name in tqdm(names):
            control = load_json_data(os.path.join(control_dir, name))
            output = load_json_data(os.path.join(output_dir, name))

            if output['responsemsg'] != 'ok':
                continue

            cp = parse_control(control) # T x CP_DIM
            ep = parse_output(output, self.EP_KEYS) # T x EP_DIM
            op = parse_output(output, self.OP_KEYS) # T x OP_DIM

            cp_dim = cp.shape[1]
            ep_dim = ep.shape[1]
            op_dim = op.shape[1]

            cp = cp.reshape(-1, 24, cp_dim)  # D x 24 x CP_DIM
            ep = ep.reshape(-1, 24, ep_dim)  # D x 24 x EP_DIM
            op = op.reshape(-1, 24, op_dim)  # D x 24 x OP_DIM

            # assumption: cp[d+1] + ep[d+1] + op[d] -> op[d+1]
            cp = cp[1:]
            ep = ep[1:]
            op_next = op[1:]
            op = op[:-1]

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

    @staticmethod
    def get_norm_data(data_dirs):
        return {
            'cp': get_control_range(data_dirs),
            'ep': get_output_range(data_dirs, ClimateDatasetDay.EP_KEYS),
            'op': get_output_range(data_dirs, ClimateDatasetDay.OP_KEYS),
        }

    @property
    def meta_data(self):
        return {
            'cp_dim': self.cp_dim,
            'ep_dim': self.ep_dim,
            'op_dim': self.op_dim,
            'ranges': self.ranges,
        }


class PlantDatasetDay(Dataset):
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
        ranges, 
        force_preprocess=False,
        data_name="plant_data_day",
    ) -> None:
        super().__init__()
        self.ranges = ranges
        self.data_name = data_name

        op_in, op_pl, op_pl_next = [], [], []
        for data_dir in data_dirs:
            data_path = f'{data_dir}/{data_name}.npz'
            if not os.path.exists(data_path) or force_preprocess:
                self.preprocess_data(data_dir)
            data = np.load(data_path)

            op_in.append(data['op_in'])
            op_pl.append(data['op_pl'])
            op_pl_next.append(data['op_pl_next'])

        self.op_in = np.concatenate(op_in, axis=0, dtype=np.float32)
        self.op_pl = np.concatenate(op_pl, axis=0, dtype=np.float32)
        self.op_pl_next = np.concatenate(op_pl_next, axis=0, dtype=np.float32)

        # self.op_in_normed = self.op_in.reshape(-1, 24 * len(self.OP_IN_KEYS))  # D x (24 x len(OP_IN_KEYS))
        # self.op_pl_normed = self.op_pl.mean(axis=1)  # D x len(OP_PL_KEYS)
        # self.op_pl_next_normed = self.op_pl_next.mean(axis=1)  # D x len(OP_PL_KEYS)

        self.op_in_dim = len(self.OP_IN_KEYS) * 24
        self.op_pl_dim = len(self.OP_PL_KEYS)

        self.op_in_normed = normalize_zero2one(self.op_in, ranges['op_in']).reshape(-1, self.op_in_dim)
        self.op_pl_normed = normalize_zero2one(self.op_pl, ranges['op_pl']).mean(axis=1)
        self.op_pl_next_normed = normalize_zero2one(self.op_pl_next, ranges['op_pl']).mean(axis=1)

    def __getitem__(self, index):
        input = (self.op_in_normed[index], self.op_pl_normed[index])
        target = self.op_pl_next_normed[index] 
        return input, target

    def __len__(self):
        return self.op_in.shape[0]

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

            op_in = parse_output(output, self.OP_IN_KEYS) # T x OP_IN_DIM
            op_pl = parse_output(output, self.OP_PL_KEYS) # T x OP_PL_DIM

            op_in = op_in.reshape(-1, 24, op_in.shape[-1])  # D x 24 x OP_IN_DIM
            op_pl = op_pl.reshape(-1, 24, op_pl.shape[-1])  # D x 24 x OP_PL_DIM

            # assumption: op_in[d+1] + op_pl[d] -> op_pl[d+1]
            op_in = op_in[1:]
            op_pl_next = op_pl[1:]
            op_pl = op_pl[:-1]

            op_in_arr.append(op_in)
            op_pl_arr.append(op_pl)
            op_pl_next_arr.append(op_pl_next)
        
        op_in_arr = np.concatenate(op_in_arr, axis=0)
        op_pl_arr = np.concatenate(op_pl_arr, axis=0)
        op_pl_next_arr = np.concatenate(op_pl_next_arr, axis=0)

        np.savez_compressed(
            f'{data_dir}/{self.data_name}.npz', 
            op_in=op_in_arr, op_pl=op_pl_arr, op_pl_next=op_pl_next_arr)

    @property
    def meta_data(self):
        return {
            'op_in_dim': self.op_in_dim,
            'op_pl_dim': self.op_pl_dim,
            'ranges': self.ranges,
        }

    @staticmethod
    def get_norm_data(norm_data_dirs):
        return {
            'op_in': get_output_range(norm_data_dirs, PlantDatasetDay.OP_IN_KEYS),
            'op_pl': get_output_range(norm_data_dirs, PlantDatasetDay.OP_PL_KEYS),
        }
