import os
import json
import datetime
import numpy as np


START_DATE = datetime.date(2021, 3, 4)

CONTROL_KEYS = [
    # simulation duration
    "simset.@endDate",

    # heating pipe settings: influence temperature
    "comp1.heatingpipes.pipe1.@maxTemp",
    "comp1.heatingpipes.pipe1.@minTemp",
    "comp1.heatingpipes.pipe1.@radiationInfluence",
    
    # greenhouse temperature settings
    "comp1.setpoints.temp.@heatingTemp",
    "comp1.setpoints.temp.@ventOffset",
    "comp1.setpoints.temp.@radiationInfluence",
    "comp1.setpoints.temp.@PbandVent",
    
    # ventliation settings: influence both temperature and humidity
    "comp1.setpoints.ventilation.@startWnd",
    "comp1.setpoints.ventilation.@winLeeMin",
    "comp1.setpoints.ventilation.@winLeeMax",
    "comp1.setpoints.ventilation.@winWndMin",
    "comp1.setpoints.ventilation.@winWndMax",
    
    # CO2 supply and density settings
    "common.CO2dosing.@pureCO2cap",
    "comp1.setpoints.CO2.@setpoint",
    "comp1.setpoints.CO2.@setpIfLamps",
    "comp1.setpoints.CO2.@doseCapacity",

    # screen settings: influence sunlight exposure and light pollution fine during night time
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

    # illumination settings: influence overall lighting time
    "comp1.illumination.lmp1.@enabled",
    "comp1.illumination.lmp1.@intensity",
    "comp1.illumination.lmp1.@hoursLight",
    "comp1.illumination.lmp1.@endTime",
    "comp1.illumination.lmp1.@maxIglob",
    "comp1.illumination.lmp1.@maxPARsum",

    # planting density for multiple growth stages: influence overall grorwing speed and final quality of the crop
    "crp_lettuce.Intkam.management.@plantDensity",
]


def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_json_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def valseq_to_scheme(vals, start_date):
    assert isinstance(vals, list) and len(vals) > 0 and len(vals) % 24 == 0

    scheme = {}
    for i in range(len(vals) // 24):
        date = start_date + datetime.timedelta(days=i)
        vals_day = vals[i*24:(i+1)*24]
        scheme[f'{date.day:02d}-{date.month:02d}'] = {str(t): vals_day[t] for t in range(24)}
    
    return scheme


class ControlParamSimple(object):
    def __init__(self, init_json_path='ClimateControlSample.json', start_date=START_DATE):
        super().__init__()
        with open(init_json_path, 'r') as f:
            self.data = json.load(f)

        assert type(start_date) == datetime.date
        self.start_date = start_date

    def dump_json(self, save_dir, save_name=None):
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = f'{hex(hash(self))}.json'
        with open(os.path.join(save_dir, save_name), 'w') as f:
            json.dump(self.data, f, indent=4)
    
    def set_value(self, key_path, value):
        keys = key_path.split('.')
        field = self.data
        for key in keys[:-1]:
            field = field[key]

        if type(value) in [int, float, bool, str, dict]:
            field[keys[-1]] = value
        elif type(value) == list:
            field[keys[-1]] = valseq_to_scheme(value, self.start_date)
        else:
            raise ValueError(f'value type of {value} not supported!')
    
    def set_endDate(self, num_days:int):
        end_date = self.start_date + datetime.timedelta(days=num_days)
        self.set_value("simset.@endDate", end_date.isoformat())

    def __repr__(self):
        return json.dumps(self.data, indent=4)

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return str(self.data).__hash__()


class ControlParams(object):

    START_DATE = datetime.date(2021, 3, 4)

    def __init__(self, init_json_path='ClimateControlSample.json'):
        super().__init__()
        with open(init_json_path, 'r') as f:
            self.data = json.load(f)

    def save_as_json(self, save_dir, save_name=None):
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = f'{hex(hash(self))}.json'
        with open(os.path.join(save_dir, save_name), 'w') as f:
            json.dump(self.data, f, indent=4)

    def set_end_date(self, duration=None):
        if duration is not None:
            end_date = self.START_DATE + datetime.timedelta(days=duration)
            self.data["simset"] = {"@endDate": end_date.strftime("%Y-%m-%d")}

    def set_heatingpipes(self, maxTemp=None, minTemp=None, radiationInfluence=None):
        if maxTemp is not None:
            self.data["comp1"]["heatingpipes"]["pipe1"]["@maxTemp"] = maxTemp
        if minTemp is not None:
            self.data["comp1"]["heatingpipes"]["pipe1"]["@minTemp"] = minTemp
        if radiationInfluence is not None:
            self.data["comp1"]["heatingpipes"]["pipe1"]["@radiationInfluence"] = radiationInfluence

    def set_temperature(self, heatingTemp=None, ventOffset=None, radiationInfluence=None, PbandVent=None):
        if heatingTemp is not None:
            self.data["comp1"]["setpoints"]["temp"]["@heatingTemp"] = heatingTemp
        if ventOffset is not None:
            self.data["comp1"]["setpoints"]["temp"]["@ventOffset"] = ventOffset
        if radiationInfluence is not None:
            self.data["comp1"]["setpoints"]["temp"]["@radiationInfluence"] = radiationInfluence
        if PbandVent is not None:
            self.data["comp1"]["setpoints"]["temp"]["@PbandVent"] = PbandVent

    def set_ventilation(self, startWnd=None, winLeeMin=None, winLeeMax=None, winWndMin=None, winWndMax=None):
        if startWnd is not None:
            self.data["comp1"]["setpoints"]["ventilation"]["@startWnd"] = startWnd
        if winLeeMin is not None:
            self.data["comp1"]["setpoints"]["ventilation"]["@winLeeMin"] = winLeeMin
        if winLeeMax is not None:
            self.data["comp1"]["setpoints"]["ventilation"]["@winLeeMax"] = winLeeMax
        if winWndMin is not None:
            self.data["comp1"]["setpoints"]["ventilation"]["@winWndMin"] = winWndMin
        if winWndMax is not None:
            self.data["comp1"]["setpoints"]["ventilation"]["@winWndMax"] = winWndMax

    def set_CO2(self, pureCO2cap=None, setpoint=None, setpIfLamps=None, doseCapacity=None):
        if pureCO2cap is not None:
            self.data["common"]["CO2dosing"]["@pureCO2cap"] = pureCO2cap
        if setpoint is not None:
            self.data["comp1"]["setpoints"]["CO2"]["@setpoint"] = setpoint
        if setpIfLamps is not None:
            self.data["comp1"]["setpoints"]["CO2"]["@setpIfLamps"] = setpIfLamps
        if doseCapacity is not None:
            self.data["comp1"]["setpoints"]["CO2"]["@doseCapacity"] = doseCapacity

    def set_illumination(self, enabled=None, intensity=None, hoursLight=None, endTime=None, maxIglob=None, maxPARsum=None):
        if enabled is not None:
            self.data["comp1"]["illumination"]["lmp1"]["@enabled"] = enabled
        if intensity is not None:
            self.data["comp1"]["illumination"]["lmp1"]["@intensity"] = intensity
        if hoursLight is not None:
            self.data["comp1"]["illumination"]["lmp1"]["@hoursLight"] = hoursLight
        if endTime is not None:
            self.data["comp1"]["illumination"]["lmp1"]["@endTime"] = endTime
        if maxIglob is not None:
            self.data["comp1"]["illumination"]["lmp1"]["@maxIglob"] = maxIglob
        if maxPARsum is not None:
            self.data["comp1"]["illumination"]["lmp1"]["@maxPARsum"] = maxPARsum

    def set_plant_density(self, plantDensity=None):
        if plantDensity is not None:
            self.data["crp_lettuce"]["Intkam"]["management"]["@plantDensity"] = plantDensity

    def set_screen(self, scr_id=1, enabled=None, material=None, ToutMax=None, closeBelow=None, closeAbove=None, lightPollutionPrevention=None):
        screen = f'scr{scr_id}'
        if enabled is not None:
            self.data["comp1"]["screens"][screen]["@enabled"] = enabled
        if material is not None:
            self.data["comp1"]["screens"][screen]["@material"] = material
        if ToutMax is not None:
            self.data["comp1"]["screens"][screen]["@ToutMax"] = ToutMax
        if closeBelow is not None:
            self.data["comp1"]["screens"][screen]["@closeBelow"] = closeBelow
        if closeAbove is not None:
            self.data["comp1"]["screens"][screen]["@closeAbove"] = closeAbove
        if lightPollutionPrevention is not None:
            self.data["comp1"]["screens"][screen]["@lightPollutionPrevention"] = lightPollutionPrevention

    def __repr__(self):
        return json.dumps(self.data, indent=4, sort_keys=True)

    def __str__(self) -> str:
        return self.data.__str__()

    def __hash__(self) -> int:
        return str(self.data).__hash__()


if __name__ == '__main__':
    vals = np.random.randint(10, 30, (24*2,)).tolist()
    # vals = [str(x) for x in vals]
    CP = ControlParamSimple()
    CP.set_endDate(2)
    CP.set_value("comp1.setpoints.temp.@heatingTemp", vals)
    print(CP)
