import random
from utils import ControlParams


class ControlParamSampleSpace(object):

    def __init__(self):
        super().__init__()
    
    def sample_control_params(self):
        duration = self.sample_duration()
        
        maxTemp = self.sample_heatingpipes_maxTemp()
        minTemp = self.sample_heatingpipes_minTemp()
        radinf_hp = self.sample_heatingpipes_radiationInfluence()
        
        heatingTemp = self.sample_temp_heatingTemp()
        ventOffset = self.sample_temp_ventOffset()
        radinf_temp = self.sample_temp_radiationInfluence()
        PbandVent = self.sample_temp_PbandVent()
        
        pureCO2cap = self.sample_CO2_pureCO2cap()
        setpoint = self.sample_CO2_setpoint()
        setpIfLamps = self.sample_CO2_setpIfLamps()
        doseCapacity = self.sample_CO2_doseCapacity()
        
        light_enabled = self.sample_illumination_enabled()
        intensity = self.sample_illumination_intensity()
        hoursLight = self.sample_illumination_hoursLight()
        endTime = self.sample_illumination_endTime()
        maxIglob = self.sample_illumination_maxIglob()
        maxPARsum = self.sample_illumination_maxPARsum()
        
        startWnd = self.sample_ventilation_startWnd()
        winLeeMin, winLeeMax = self.sample_ventilation_winLeeMinMax()
        winWndMin, winWndMax = self.sample_ventilation_winWndMinMax()
        
        plantDensity = self.sample_plantDensity()
        
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

        CP = ControlParams(initialize_with_sample=False)
        CP.set_end_date(duration=duration)
        CP.set_heatingpipes(maxTemp=maxTemp, minTemp=minTemp, radiationInfluence=radinf_hp)
        CP.set_temperature(heatingTemp=heatingTemp, ventOffset=ventOffset, radiationInfluence=radinf_temp, PbandVent=PbandVent)
        CP.set_CO2(pureCO2cap=pureCO2cap, setpoint=setpoint, setpIfLamps=setpIfLamps, doseCapacity=doseCapacity)
        CP.set_illumination(enabled=light_enabled, intensity=intensity, hoursLight=hoursLight, endTime=endTime, maxIglob=maxIglob, maxPARsum=maxPARsum)
        CP.set_ventilation(startWnd=startWnd, winLeeMin=winLeeMin, winLeeMax=winLeeMax, winWndMin=winWndMin, winWndMax=winWndMax)
        CP.set_plant_density(plantDensity=plantDensity)
        CP.set_screen(scr_id=1, enabled=scr1_enabled, material=scr1_material, ToutMax=scr1_ToutMax, closeBelow=scr1_closeBelow, closeAbove=scr1_closeAbove, lightPollutionPrevention=scr1_lightPollutionPrevention)
        CP.set_screen(scr_id=2, enabled=scr2_enabled, material=scr2_material, ToutMax=scr2_ToutMax, closeBelow=scr2_closeBelow, closeAbove=scr2_closeAbove, lightPollutionPrevention=scr2_lightPollutionPrevention)

        return CP

    def sample_duration(self, min=30, max=50):
        duration = random.randrange(min, max, step=2)
        return duration

    def sample_heatingpipes_maxTemp(self, min=45, max=75):
        maxTemp = random.randrange(min, max, step=3)
        return maxTemp
    
    def sample_heatingpipes_minTemp(self, min=0, max=45):
        minTemp = random.randrange(min, max, step=3)
        return minTemp

    def sample_heatingpipes_radiationInfluence(self):
        # TODO: need to design sample space
        # radinf = "100 300"
        radinf = "0"
        return radinf

    def sample_temp_heatingTemp(self):
        # TODO: need to design better sample space
        # temp_high = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        # temp_low = [8, 10, 12, 14, 16]
        temp = [10, 12, 14, 16, 18, 20, 22, 24]
        schemes = [
            # {"01-01": {"r-1": t_low,"r+1": t_high, "s-1": t_high, "s+1": t_low}} for t_low in temp_low for t_high in temp_high if t_low <= t_high
            {"01-01": {"8.0":t}} for t in temp
        ]
        heatingTemp = random.choice(schemes)
        return heatingTemp

    def sample_temp_ventOffset(self):
        # TODO: need to design sample space
        ventOffset = {"01-01": {"0": 2}}
        return ventOffset

    def sample_temp_radiationInfluence(self):
        # TODO: need to design sample space
        radinf = {"01-01": "50 150 1"}
        return radinf

    def sample_temp_PbandVent(self):
        # TODO: need to design sample space
        PbandVent="0 10; 20 5"
        return PbandVent

    def sample_CO2_pureCO2cap(self, min=50, max=200):
        pureCO2cap = random.randrange(min, max, step=10)
        return pureCO2cap

    def sample_CO2_setpoint(self, min=400, max=1200):
        setpoint = random.randrange(min, max, step=100)
        return setpoint

    def sample_CO2_setpIfLamps(self, min=700, max=900):
        setpIfLamps = random.randrange(min, max, step=10)
        return setpIfLamps

    def sample_CO2_doseCapacity(self):
        # TODO: need to design sample space
        doseCapacity = "100"
        return doseCapacity

    def sample_illumination_enabled(self):
        # TODO: need to design sample space
        enabled = True
        return enabled

    def sample_illumination_intensity(self, min=50, max=200):
        intensity = random.randrange(min, max, step=10)
        return intensity

    def sample_illumination_hoursLight(self, min=12, max=20):
        hoursLight = random.randrange(min, max, step=2)
        return hoursLight

    def sample_illumination_endTime(self, min=18, max=24):
        endTime = random.randrange(min, max, step=2)
        return endTime

    def sample_illumination_maxIglob(self, min=150, max=300):
        maxIglob = random.randrange(min, max, step=10)
        return maxIglob

    def sample_illumination_maxPARsum(self, min=15, max=40):
        maxPARsum = random.randrange(min, max, step=2)
        return maxPARsum

    def sample_ventilation_startWnd(self, min=0, max=50):
        startWnd = random.randrange(min, max, step=5)
        return startWnd

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

    def sample_plantDensity(self):
        # TODO: need to design better sample space
        schemes = [
            "1 90; 7 60; 14 40; 21 30; 28 20; 34 15",
            "1 80; 10 40; 20 30; 25 20; 30 10",
        ]
        plantDensity = random.choice(schemes)
        return plantDensity

    def sample_screen_enabled(self):
        # TODO: need to design sample space for more fine-grained control
        enabled = random.choice([True, False])
        return enabled

    def sample_screen_material_lpp(self):
        types = [
            "scr_Blackout.par",
            "scr_Transparent.par",
            "scr_Shade.par"
        ]
        material = random.choice(types)
        lightPollutionPrevention = material != "scr_Transparent.par"
        return material, lightPollutionPrevention

    def sample_screen_ToutMax(self, min=-20, max=30):
        ToutMax = random.randrange(min, max, step=2)
        return ToutMax

    def sample_screen_closeBelow(self, min=5, max=100):
        # TODO: need to design boarder-line based sample space 
        closeBelow = random.randrange(min, max, step=5)
        return closeBelow

    def sample_screen_closeAbove(self, min=1000, max=1400):
        # TODO: need to design boarder-line based sample space 
        closeAbove = random.randrange(min, max, step=50)
        return closeAbove


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
        control.save_as_json(f'{args.control_json_dir}/{hashcode}')
