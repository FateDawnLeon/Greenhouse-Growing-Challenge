import random
from utils import ControlParams
import datetime

class ControlParamSampleSpace(object):

    def __init__(self):
        super().__init__()
    
    def sample_control_params(self):
        startdate, startdate_gap = self.sample_startdate()
        enddate, duration = self.sample_enddate(startdate_gap)
        
        temp_maxTemp = self.sample_heatingpipes_maxTemp()
        temp_minTemp = self.sample_heatingpipes_minTemp()
        temp_radinf_hp = self.sample_heatingpipes_radiationInfluence()
        temp_heatingTemp = self.sample_temp_heatingTemp()
        temp_ventOffset = self.sample_temp_ventOffset()
        temp_radinf = self.sample_temp_radiationInfluence()
        temp_PbandVent = self.sample_temp_PbandVent()
        
        CO2_purecap = self.sample_CO2_pureCO2cap()
        CO2_setpoint = self.sample_CO2_setpoint()
        CO2_setpIfLamps = self.sample_CO2_setpIfLamps()
        CO2_doseCapacity = self.sample_CO2_doseCapacity()
        
        # light_enabled = self.sample_illumination_enabled()
        light_intensity = self.sample_illumination_intensity()
        light_hours = self.sample_illumination_hoursLight()
        light_endTime = self.sample_illumination_endTime()
        light_maxIglob = self.sample_illumination_maxIglob()
        light_maxPARsum = self.sample_illumination_maxPARsum()
        
        vent_startWnd = self.sample_ventilation_startWnd()
        vent_winLeeMin, vent_winLeeMax = self.sample_ventilation_winLeeMinMax()
        vent_winWndMin, vent_winWndMax = self.sample_ventilation_winWndMinMax()
        
        plantDensity = self.sample_plantDensity(duration)
        
        scr1_enabled = self.sample_screen_enabled()
        scr1_material, scr1_lightPollutionPrevention = self.sample_screen_material_lpp()
        scr1_ToutMax = self.sample_screen_ToutMax()
        scr1_closeBelow = self.sample_screen_closeBelow()
        scr1_closeAbove = self.sample_screen_closeAbove()
        
        scr2_enabled = self.sample_screen_enabled()
        scr2_material, scr2_lightPollutionPrevention = self.sample_screen_material_lpp()
        scr2_ToutMax = self.sample_screen_ToutMax()
        scr2_closeBelow = self.sample_screen_closeBelow()
        scr2_closeAbove = self.sample_screen_closeAbove()

        CP = ControlParams(start_date = startdate)
        CP.set_value("simset.@startDate", startdate.isoformat())
        CP.set_value("simset.@endDate", enddate.isoformat())

        CP.set_value("common.CO2dosing.@pureCO2cap", CO2_purecap)

        CP.set_value("comp1.heatingpipes.pipe1.@maxTemp", temp_maxTemp)
        CP.set_value("comp1.heatingpipes.pipe1.@minTemp", temp_minTemp)
        CP.set_value("comp1.heatingpipes.pipe1.@radiationInfluence", temp_radinf_hp)

        CP.set_value("comp1.screens.scr1.@enabled", scr1_enabled)
        CP.set_value("comp1.screens.scr1.@material", scr1_material)
        CP.set_value("comp1.screens.scr1.@closeBelow", scr1_closeBelow)
        CP.set_value("comp1.screens.scr1.@closeAbove", scr1_closeAbove)
        CP.set_value("comp1.screens.scr1.@ToutMax", scr1_ToutMax)
        CP.set_value("comp1.screens.scr1.@lightPollutionPrevention", scr1_lightPollutionPrevention)

        CP.set_value("comp1.screens.scr2.@enabled", scr2_enabled)
        CP.set_value("comp1.screens.scr2.@material", scr2_material)
        CP.set_value("comp1.screens.scr2.@closeBelow", scr2_closeBelow)
        CP.set_value("comp1.screens.scr2.@closeAbove", scr2_closeAbove)
        CP.set_value("comp1.screens.scr2.@ToutMax", scr2_ToutMax)
        CP.set_value("comp1.screens.scr2.@lightPollutionPrevention", scr2_lightPollutionPrevention)

        CP.set_value("comp1.illumination.lmp1.@enabled", light_intensity > 0)
        CP.set_value("comp1.illumination.lmp1.@intensity", light_intensity)
        CP.set_value("comp1.illumination.lmp1.@hoursLight", light_hours)
        CP.set_value("comp1.illumination.lmp1.@endTime", light_endTime)
        CP.set_value("comp1.illumination.lmp1.@maxIglob", light_maxIglob)
        CP.set_value("comp1.illumination.lmp1.@maxPARsum", light_maxPARsum)

        CP.set_value("comp1.setpoints.temp.@heatingTemp", temp_heatingTemp)
        CP.set_value("comp1.setpoints.temp.@radiationInfluence", temp_radinf)
        CP.set_value("comp1.setpoints.temp.@ventOffset", temp_ventOffset)
        CP.set_value("comp1.setpoints.temp.@PbandVent", temp_PbandVent)

        CP.set_value("comp1.setpoints.CO2.@setpoint", CO2_setpoint)
        CP.set_value("comp1.setpoints.CO2.@setpIfLamps", CO2_setpIfLamps)
        CP.set_value("comp1.setpoints.CO2.@doseCapacity", CO2_doseCapacity)

        CP.set_value("comp1.setpoints.ventilation.@winLeeMin", vent_winLeeMin)
        CP.set_value("comp1.setpoints.ventilation.@winLeeMax", vent_winLeeMax)
        CP.set_value("comp1.setpoints.ventilation.@winWndMin", vent_winWndMin)
        CP.set_value("comp1.setpoints.ventilation.@winWndMax", vent_winWndMax)
        CP.set_value("comp1.setpoints.ventilation.@startWnd", vent_startWnd)

        CP.set_value("crp_lettuce.Intkam.management.@plantDensity", plantDensity)

        return CP

    def sample_startdate(self, min=0, max=66):
        startdate_gap = random.randrange(min, max)
        startdate =  datetime.date(2021, 2, 25) + datetime.timedelta(days=startdate_gap)
        return startdate, startdate_gap

    def sample_enddate(self, startdate_gap, min = 30, max = 50):
        duration = random.randrange(min, max)
        duration = 66 - startdate_gap if (startdate_gap+duration > 66) else duration
        startdate_gap = startdate_gap + duration
        enddate = datetime.date(2021, 2, 25) + datetime.timedelta(days=startdate_gap)
        return enddate, duration

    def sample_heatingpipes_maxTemp(self, min=45, max=75):
        return random.randrange(min, max, step=3)
    
    def sample_heatingpipes_minTemp(self, min=0, max=45):
        return random.randrange(min, max, step=3)

    def sample_heatingpipes_radiationInfluence(self):
        return "0"

    def sample_temp_heatingTemp(self, min_night=2, max_night=10, min_day=15, max_day=25):
        heatingTemp_night = random.randrange(min_night, max_night, step=1)
        heatingTemp_day = random.randrange(min_day, max_day, step=1)

        heating_temp = {
            "01-01": {
                "r": heatingTemp_night,
                "8": heatingTemp_day, 
                "17": heatingTemp_day, 
                "s": heatingTemp_night
            }
        }
        return heating_temp

    def sample_temp_ventOffset(self):
        return {
					"01-01" : {"r" : 1,"r+1" : 2, "s" : 2, "s+1" : 1}
				}

    def sample_temp_radiationInfluence(self):
        return "200 400 2"

    def sample_temp_PbandVent(self):
        return "0 10; 20 5"

    def sample_CO2_pureCO2cap(self, min=260, max=280):
        return random.randrange(min, max, step=5)

    def sample_CO2_setpoint(self, min_night=400, max_night=600, min_day=1100, max_day=1200):
        CO2_setpoint_night = random.randrange(min_night, max_night, step=10)
        CO2_setpoint_day = random.randrange(min_day, max_day, step=10)

        CO2_setpoint = {
            "01-01": {
                "r": CO2_setpoint_night, 
                "8": CO2_setpoint_day,
                "17": CO2_setpoint_day, 
                "s": CO2_setpoint_night,
            }
        }
        return CO2_setpoint

    def sample_CO2_setpIfLamps(self, min=1100, max=1200):
        return random.randrange(min, max, step=10)

    def sample_CO2_doseCapacity(self):
        return "100"

    def sample_illumination_enabled(self):
        return True

    def sample_illumination_intensity(self, min=0, max=10):
        return random.randrange(min, max, step=1)

    def sample_illumination_hoursLight(self, min=10, max=15):
        return random.randrange(min, max, step=1)

    def sample_illumination_endTime(self, min=17, max=21):
        return random.randrange(min, max, step=1)

    def sample_illumination_maxIglob(self, min=100, max=500):
        return random.randrange(min, max, step=50)

    def sample_illumination_maxPARsum(self, min=15, max=50):
        return random.randrange(min, max, step=2)

    def sample_ventilation_startWnd(self, min=50, max=55):
        return random.randrange(min, max, step=1)

    def sample_ventilation_winLeeMinMax(self):
        # TODO: need to design sample space for more fine-grained control
        # winLeeMin = random.randint(0, 100)
        # winLeeMax = random.randint(max(winLeeMin, 30), 100)
        winLeeMin = 0
        winLeeMax = 100
        return winLeeMin, winLeeMax

    def sample_ventilation_winWndMinMax(self):
        # TODO: need to design sample space for more fine-grained control
        # winWndMin = random.randint(0, 100)
        # winWndMax = random.randint(max(winWndMin, 30), 100)
        winWndMin = 0
        winWndMax = 100
        return winWndMin, winWndMax

    def sample_plantDensity(self, duration):
        import numpy as np
        start_density_range = np.arange(80, 91, 5)  # 80,85,90
        end_density_range = np.arange(5, 16, 5)  # 5,10,15
        skip_day_range = np.arange(5, 11, 1)  # 5,6,7,8,9,10
        change_density_range = np.arange(5, 36, 5)  # 5,10,15,20,25,30,35

        control_densitys = [
            "1 80; 11 45; 19 25; 27 15",
            "1 90; 7 60; 14 40; 21 30; 28 20; 34 15",
            "1 80; 9 50; 14 25; 20 20; 27 15",
            "1 80; 12 45; 20 25; 27 20; 35 10",  # from email
            "1 80; 10 40; 20 30; 25 20; 30 10",  # from control sample
            "1 80; 10 55; 15 40; 20 25; 25 20; 31 15",  # from C15TEST
            "1 85; 7 50; 15 30; 25 20; 33 15",  # from D1 test on sim C
        ]

        for i in range(500):
            # "1 90; 7 60; 14 40; 21 30; 28 20; 34 15"
            start_density = random.choice(start_density_range)
            end_density = random.choice(end_density_range)

            days = 1
            density = start_density

            skip_days = []
            change_densitys = []
            while True:
                skip_day = random.choice(skip_day_range)
                change_density = random.choice(change_density_range)
                days += skip_day
                if days>duration: break
                density = density - change_density
                if density<end_density: break
                skip_days.append(skip_day)
                change_densitys.append(change_density)
            change_densitys.sort(reverse=True)

            days = 1
            density = start_density
            control_density =  f'{days} {density}'
            for i in range(len(skip_days)):
                days += skip_days[i]
                density = density - change_densitys[i]
                control_density = f'{control_density}; {days} {density}'

            if density in end_density_range:
                control_densitys.append(control_density)

        return random.choice(control_densitys)

    def sample_screen_enabled(self):
        return random.choice([True, False])

    def sample_screen_material_lpp(self):
        types = [
            "scr_Blackout.par",
            "scr_Transparent.par",
            "scr_Shade.par"
        ]
        material = random.choice(types)
        lightPollutionPrevention = material != "scr_Transparent.par"
        return material, lightPollutionPrevention

    def sample_screen_ToutMax(self, min=4, max=6):
        return random.randrange(min, max, step=1)

    def sample_screen_closeBelow(self, min=5, max=100):
        return random.randrange(min, max, step=5)

    def sample_screen_closeAbove(self, min=1000, max=1400):
        return random.randrange(min, max, step=50)


if __name__ == '__main__':
    import os
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num-controls', type=int)
    parser.add_argument('-D', '--control-json-dir', type=str, default='control_jsons')
    args = parser.parse_args()

    os.makedirs(args.control_json_dir, exist_ok=True)
    control_json_names = os.listdir(args.control_json_dir)

    SP = ControlParamSampleSpace()
    
    for _ in tqdm(range(args.num_controls)):
        while True:
            control = SP.sample_control_params()
            hashcode = hex(hash(control))
            if f'{hashcode}.json' not in control_json_names:
                break
        control.dump_json(save_dir = f'{args.control_json_dir}', save_name = f'{hashcode}.json')
