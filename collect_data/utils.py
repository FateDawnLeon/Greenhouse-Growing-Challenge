import os
import json
import datetime


control_params_path = [
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
    CP = ControlParams()

    # default settings from the sample json file
    CP.set_end_date(duration=40)
    CP.set_heatingpipes(
        maxTemp=60,
        minTemp=0, 
        radiationInfluence="100 300"
    )
    CP.set_temperature(
        heatingTemp={"01-01": {"r-1": 10,"r+1": 12, "s-1": 12, "s+1": 10}}, 
        ventOffset={"01-01" : {"0" : 2}},
        radiationInfluence={"01-01": "50 150 1"},
        PbandVent="0 10; 20 5"
    )
    CP.set_CO2(
        pureCO2cap=100, 
        setpoint=600, 
        setpIfLamps=800, 
        doseCapacity="100"
    )
    CP.set_illumination(
        enabled=True, 
        intensity=80, 
        hoursLight=14, 
        endTime=18, 
        maxIglob=100, 
        maxPARsum=50
    )
    CP.set_ventilation(
        startWnd=50, 
        winLeeMin=0, 
        winLeeMax=100, 
        winWndMin=0, 
        winWndMax=100
    )
    CP.set_plant_density(
        plantDensity="1 80; 10 40; 20 30; 25 20; 30 10"
    )
    CP.set_screen(
        scr_id=1, 
        enabled=True,
        material="scr_Blackout.par",
        ToutMax=8,
        closeBelow=5,
        closeAbove=1200,
        lightPollutionPrevention=True
    )
    CP.set_screen(
        scr_id=2, 
        enabled=False, 
        material="scr_Transparent.par", 
        ToutMax=12, 
        closeBelow="0 100; 10 5", 
        closeAbove=1000,
        lightPollutionPrevention=False
    )

    print(CP)
