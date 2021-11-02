import random

from data.random_hackathon.get_time import get_sun_rise_and_set
from utils import ControlParams
import datetime
from astral.geocoder import lookup, database

BO_NUMBER = 100
RL_NUMBER = 1000

class ControlParamSampleSpace(object):

    def __init__(self):
        super().__init__()
        self.startdate_gap = random.randrange(0, 31)
        self.CO2_purecap = round(random.uniform(100, 500),1)
        self.scr1_enabled = random.choice([True, False])
        self.scr1_material = random.choice(["scr_Blackout.par", "scr_Transparent.par", "scr_Shade.par"])
        self.scr2_enabled = random.choice([True, False])
        self.scr2_material = random.choice(["scr_Blackout.par", "scr_Transparent.par", "scr_Shade.par"])
        self.light_intensity = round(random.uniform(0, 500),1)
        self.light_maxIglob = round(random.uniform(0, 500),1)
        self.start_density = random.choice([90, 85, 80])
        self.change_amount = round(random.uniform(1, 35),1)

        # self.startdate_gap = 7
        # self.CO2_purecap = 280
        # self.scr1_enabled = True
        # self.scr1_material = "scr_Blackout.par"
        # self.scr2_enabled =  False
        # self.scr2_material = "scr_Transparent.par"
        # self.light_intensity = 3.0
        # self.light_maxIglob = 500.0
        # self.change_amount = 22

    def sample_control_params(self):
        startdate, startdate_gap = self.sample_startdate()
        enddate, duration = self.sample_enddate(startdate_gap)
        self.start = startdate
        self.end = enddate

        # light_enabled = self.sample_illumination_enabled()
        light_hours = self.sample_illumination_hoursLight()
        self.hours =  light_hours
        light_endTime = self.sample_illumination_endTime()
        self.endtime = light_endTime
        light_intensity = self.sample_illumination_intensity()
        light_maxIglob = self.sample_illumination_maxIglob()
        light_maxPARsum = self.sample_illumination_maxPARsum()
        
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
        

        
        vent_startWnd = self.sample_ventilation_startWnd()
        vent_winLeeMin, vent_winLeeMax = self.sample_ventilation_winLeeMinMax()
        vent_winWndMin, vent_winWndMax = self.sample_ventilation_winWndMinMax()
        
        plantDensity = self.sample_plantDensity(duration)
        
        scr1_enabled = self.sample_screen1_enabled()
        scr1_material, scr1_lightPollutionPrevention = self.sample_screen1_material_lpp()
        scr1_ToutMax = self.sample_screen_ToutMax()
        scr1_closeBelow = self.sample_screen_closeBelow()
        scr1_closeAbove = self.sample_screen_closeAbove()
        
        scr2_enabled = self.sample_screen2_enabled()
        scr2_material, scr2_lightPollutionPrevention = self.sample_screen2_material_lpp()
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
        CP.set_value("comp1.illumination.lmp1.@hoursLight", self.hours)
        CP.set_value("comp1.illumination.lmp1.@endTime", self.endtime)
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

    # 31 = 66 (latest end - ealiest start)- 35 (shortest period)
    def sample_startdate(self, min=0, max=31):
        startdate_gap = self.startdate_gap
        # startdate_gap = 17
        startdate =  datetime.date(2021, 2, 25) + datetime.timedelta(days=startdate_gap)
        return startdate, startdate_gap

    def sample_enddate(self, startdate_gap, min = 35, max = 45):
        duration = random.randrange(min, max)
        duration = 66 - startdate_gap if (startdate_gap+duration > 66) else duration
        startdate_gap = startdate_gap + duration
        enddate = datetime.date(2021, 2, 25) + datetime.timedelta(days=startdate_gap)
        return enddate, duration

    # def sample_heatingpipes_maxTemp(self, min=45, max=75):
        # return random.randrange(min, max, step=3)
    def sample_heatingpipes_maxTemp(self):
        return 60
    
    # def sample_heatingpipes_minTemp(self, min=0, max=45):
    #     return random.randrange(min, max, step=3)
    def sample_heatingpipes_minTemp(self):
        return 0

    def sample_heatingpipes_radiationInfluence(self):
        return "100 300"

    def sample_temp_heatingTemp(self, min_night=5, max_night=15, min_day=15, max_day=30):
        start = self.start
        end = self.end
        num_days = (end - start).days

        heatingTemp = {}
        for d in range(num_days):
            cur = start + datetime.timedelta(days=d)
            key = "{:02d}-{:02d}".format(cur.day, cur.month)
            heatingTemp_night = round(random.uniform(min_night, max_night),1)
            heatingTemp_day = round(random.uniform(min_day, max_day),1)
            city = lookup("Amsterdam", database())
            starttime_a = get_sun_rise_and_set(cur, city)[0]
            starttime_b = float(self.endtime.get(key)) - float(self.hours.get(key))
            starttime = min(starttime_a, starttime_b)
            endtime= float(self.endtime.get(key))
            # endtime = get_sun_rise_and_set(cur, city)[1]
            heatingTemp[key] =  {
                str(starttime): heatingTemp_night,
                str(starttime+1): heatingTemp_day,
                str(endtime-1): heatingTemp_day,
                str(endtime): heatingTemp_night
            }
        return heatingTemp


    def sample_temp_ventOffset(self, min=0, max=5):
        start = self.start
        end = self.end
        num_days = (end - start).days

        ventOffset = {}
        for d in range(num_days):
            cur = start + datetime.timedelta(days=d)
            key = "{:02d}-{:02d}".format(cur.day, cur.month)
            s = round(random.uniform(min, max),1)
            ventOffset[key] = s

        return ventOffset


    def sample_temp_radiationInfluence(self):
        return "50 150 1"

    def sample_temp_PbandVent(self):
        return "0 10; 20 5"

    # def sample_CO2_pureCO2cap(self, min=260, max=280):
    #     return random.randrange(min, max, step=5)
    def sample_CO2_pureCO2cap(self):
         return self.CO2_purecap

    def sample_CO2_setpoint(self, min_night=0, max_night=400, min_day=400, max_day=1200):
        start = self.start
        end = self.end
        num_days = (end - start).days

        CO2_setpoint = {}
        for d in range(num_days):
            cur = start + datetime.timedelta(days=d)
            key = "{:02d}-{:02d}".format(cur.day, cur.month)
            CO2_setpoint_night = round(random.uniform(min_night, max_night),1)
            CO2_setpoint_day = round(random.uniform(min_day, max_day),1)
            city = lookup("Amsterdam", database())
            starttime_a = get_sun_rise_and_set(cur, city)[0]
            starttime_b = float(self.endtime.get(key)) - float(self.hours.get(key))
            starttime = min(starttime_a, starttime_b)
            endtime = self.endtime.get(key)
            # endtime = get_sun_rise_and_set(cur, city)[1]
            CO2_setpoint[key] =  {
                str(starttime): CO2_setpoint_night,
                str(starttime+1): CO2_setpoint_day,
                str(endtime-1): CO2_setpoint_day,
                str(endtime): CO2_setpoint_night
            }
        return CO2_setpoint




    # def sample_CO2_setpIfLamps(self, min=1100, max=1200):
    #     return random.randrange(min, max, step=10)
    def sample_CO2_setpIfLamps(self):
        return 0

    def sample_CO2_doseCapacity(self):
        return 100

    def sample_illumination_enabled(self):
        return True

    # def sample_illumination_intensity(self, min=0, max=10):
    #     return random.randrange(min, max, step=1)
    def sample_illumination_intensity(self):
        return self.light_intensity

    def sample_illumination_hoursLight(self, min=0, max=18):
        start = self.start
        end = self.end
        num_days = (end - start).days

        hoursLight = {}
        for d in range(num_days):
            cur = start + datetime.timedelta(days=d)
            key = "{:02d}-{:02d}".format(cur.day, cur.month)
            s = round(random.uniform(min, max),1)
            hoursLight[key] = s

        return hoursLight


    def sample_illumination_endTime(self, min=18, max=20):
        start = self.start
        end = self.end
        num_days = (end - start).days

        endTime = {}
        for d in range(num_days):
            cur = start + datetime.timedelta(days=d)
            key = "{:02d}-{:02d}".format(cur.day, cur.month)
            s = round(random.uniform(min, max),1)
            endTime[key] = s

        return endTime

    # RL search for illumination_maxIglob
    # def sample_illumination_maxIglob(self, min=100, max=800):
    #     start = self.start
    #     end = self.end
    #     num_days = (end - start).days
    #
    #     maxIglob = {}
    #     for d in range(num_days):
    #         cur = start + datetime.timedelta(days=d)
    #         key = "{:02d}-{:02d}".format(cur.day, cur.month)
    #         s = random.randrange(min, max)
    #         maxIglob[key] = s
    #
    #     return maxIglob

    # Change to BO search for illumination_maxIglob
    def sample_illumination_maxIglob(self, min=0, max=500):
        return self.light_maxIglob

    # def sample_illumination_maxPARsum(self, min=15, max=50):
    #     return random.randrange(min, max, step=2)
    def sample_illumination_maxPARsum(self):
        return 50

    # def sample_ventilation_startWnd(self, min=50, max=55):
    #     return random.randrange(min, max, step=1)
    def sample_ventilation_startWnd(self, min=0, max=100):
        start = self.start
        end = self.end
        num_days = (end - start).days

        ventilation_startWnd = {}
        for d in range(num_days):
            cur = start + datetime.timedelta(days=d)
            key = "{:02d}-{:02d}".format(cur.day, cur.month)
            s = round(random.uniform(min, max),1)
            ventilation_startWnd[key] = s

        return ventilation_startWnd

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

    # randomly generate a list and choose from list
    # def sample_plantDensity(self, duration):
        # import numpy as np
        # start_density_range = np.arange(80, 91, 5)  # 80,85,90
        # end_density_range = np.arange(5, 16, 5)  # 5,10,15
        # skip_day_range = np.arange(5, 11, 1)  # 5,6,7,8,9,10
        # change_density_range = np.arange(5, 36, 5)  # 5,10,15,20,25,30,35
        #
        # control_densitys = [
        #     "1 80; 11 45; 19 25; 27 15",
        #     "1 90; 7 60; 14 40; 21 30; 28 20; 34 15",
        #     "1 80; 9 50; 14 25; 20 20; 27 15",
        #     "1 80; 12 45; 20 25; 27 20; 35 10",  # from email
        #     "1 80; 10 40; 20 30; 25 20; 30 10",  # from control sample
        #     "1 80; 10 55; 15 40; 20 25; 25 20; 31 15",  # from C15TEST
        #     "1 85; 7 50; 15 30; 25 20; 33 15",  # from D1 test on sim C
        # ]
        #
        # for i in range(500):
        #     # "1 90; 7 60; 14 40; 21 30; 28 20; 34 15"
        #     start_density = random.choice(start_density_range)
        #     end_density = random.choice(end_density_range)
        #
        #     days = 1
        #     density = start_density
        #
        #     skip_days = []
        #     change_densitys = []
        #     while True:
        #         skip_day = random.choice(skip_day_range)
        #         change_density = random.choice(change_density_range)
        #         days += skip_day
        #         if days>duration: break
        #         density = density - change_density
        #         if density<end_density: break
        #         skip_days.append(skip_day)
        #         change_densitys.append(change_density)
        #     change_densitys.sort(reverse=True)
        #
        #     days = 1
        #     density = start_density
        #     control_density =  f'{days} {density}'
        #     for i in range(len(skip_days)):
        #         days += skip_days[i]
        #         density = density - change_densitys[i]
        #         control_density = f'{control_density}; {days} {density}'
        #
        #     if density in end_density_range:
        #         control_densitys.append(control_density)

        # return random.choice(control_densitys)

    # density decrease everyday by change_amount with probability=prob
    def sample_plantDensity(self, duration):
        import numpy as np
        start_density = self.start_density
        density_min = 5
        change_amount = self.change_amount
        change_prob = 0.1

        start = self.start
        end = self.end
        num_days = (end - start).days

        density = [start_density]
        for i in range(num_days-1):
            if random.uniform(0,1) < change_prob:
                next_density = density[i] - change_amount
                if next_density < density_min: break
                density.append(next_density)
            else:
                next_density = density[i]
                density.append(next_density)

        days=1
        control_density = f'{days} {start_density}'
        for i in range(len(density) - 1):
            days += 1
            if density[i + 1] < density[i]:
                control_density = f'{control_density}; {days} {density[days-1]}'

        return control_density



    def sample_screen1_enabled(self):
        # return random.choice([True, False])
        return self.scr1_enabled

    def sample_screen2_enabled(self):
        # return random.choice([True, False])
        return self.scr2_enabled

    def sample_screen1_material_lpp(self):
        types = [
            "scr_Blackout.par",
            "scr_Transparent.par",
            "scr_Shade.par"
        ]
        # material = random.choice(types)
        material = self.scr1_material
        lightPollutionPrevention = material != "scr_Transparent.par"
        return material, lightPollutionPrevention

    def sample_screen2_material_lpp(self):
        types = [
            "scr_Blackout.par",
            "scr_Transparent.par",
            "scr_Shade.par"
        ]
        # material = random.choice(types)
        material = self.scr2_material
        lightPollutionPrevention = material == "scr_Blackout.par"
        return material, lightPollutionPrevention

    def sample_screen_ToutMax(self, min=-20, max=30):
        start = self.start
        end = self.end
        num_days = (end - start).days

        screen_ToutMax = {}
        for d in range(num_days):
            cur = start + datetime.timedelta(days=d)
            key = "{:02d}-{:02d}".format(cur.day, cur.month)
            s = round(random.uniform(min, max),1)
            screen_ToutMax[key] = s

        return screen_ToutMax

    def sample_screen_closeBelow(self, min=0, max=200):
        start = self.start
        end = self.end
        num_days = (end - start).days

        screen_closeBelow = {}
        for d in range(num_days):
            cur = start + datetime.timedelta(days=d)
            key = "{:02d}-{:02d}".format(cur.day, cur.month)
            s = round(random.uniform(min, max),1)
            screen_closeBelow[key] = s

        return screen_closeBelow

    def sample_screen_closeAbove(self, min=500, max=1500):
        start = self.start
        end = self.end
        num_days = (end - start).days

        screen_closeAbove = {}
        for d in range(num_days):
            cur = start + datetime.timedelta(days=d)
            key = "{:02d}-{:02d}".format(cur.day, cur.month)
            s = round(random.uniform(min, max),1)
            screen_closeAbove[key] = s

        return screen_closeAbove



if __name__ == '__main__':
    import os
    import argparse
    from tqdm import tqdm

    # BO Big Loop
    for i in range(BO_NUMBER):
        parser = argparse.ArgumentParser()

        # parser.add_argument('-N', '--num-controls', type=int)
        parser.add_argument('-N', '--num-controls', type=int, default= RL_NUMBER)
        parser.add_argument('-D', '--control-json-dir', type=str, default='controls')
        parser.add_argument('-G', '--group', type=str, default= f'group_{i}')
        args = parser.parse_args()

        os.makedirs(f'collected_data/{args.group}', exist_ok=True)
        os.makedirs(f'collected_data/{args.group}/{args.control_json_dir}', exist_ok=True)
        control_json_names = os.listdir(f'collected_data/{args.group}/{args.control_json_dir}')

        SP = ControlParamSampleSpace()

        # Save BO file
        BO = SP.sample_control_params()
        BO.dump_json(save_dir=f'collected_data/{args.group}', save_name=f'BO.json')


        for _ in tqdm(range(args.num_controls)):
            while True:
                control = SP.sample_control_params()
                hashcode = hex(hash(control))
                if f'{hashcode}.json' not in control_json_names:
                    break
            control.dump_json(save_dir = f'collected_data/{args.group}/{args.control_json_dir}', save_name = f'{hashcode}.json')

