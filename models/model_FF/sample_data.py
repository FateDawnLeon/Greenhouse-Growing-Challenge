import random
from control_param import ControlParamSimple


def sample_CP_random():
    def sample_bool():
        return random.random() > 0.5
    
    def sample_int(low, high):
        return random.randint(low, high)

    def sample_material():
        return random.choice(['scr_Transparent.par', 'scr_Shade.par', 'scr_Blackout.par'])

    def sample_line(r_x1, r_y1, r_x2, r_y2):
        x1 = sample_int(*r_x1)
        x2 = sample_int(*r_x2)
        y1 = sample_int(*r_y1)
        y2 = sample_int(*r_y2)
        return f"{x1} {y1}; {x2} {y2}"

    def val_seq(func, args_val, num_steps):
        return [func(*args_val) for _ in range(num_steps)]

    def sample_screen(CP, id):
        key_prefix = f"comp1.screens.scr{id}"
        enabled = sample_bool()
        CP.set_value(f"{key_prefix}.@enabled", enabled)
        if enabled:
            CP.set_value(f"{key_prefix}.@material", sample_material())
            CP.set_value(f"{key_prefix}.@ToutMax", val_seq(sample_int, (-20, 30), num_hours))
            CP.set_value(f"{key_prefix}.@closeBelow", val_seq(sample_line, [(0,10), (50,150), (10,30), (0,50)], num_hours))
            CP.set_value(f"{key_prefix}.@closeAbove", val_seq(sample_int, (800, 1400), num_hours))
            CP.set_value(f"{key_prefix}.@lightPollutionPrevention", True)

    def sample_doseCap(r_y1, r_y2, r_y3):
        x1 = sample_int(0, 33)
        x2 = sample_int(34, 66)
        x3 = sample_int(67, 100)
        y1 = sample_int(*r_y1)
        y2 = sample_int(*r_y2)
        y3 = sample_int(*r_y3)
        return f"{x1} {y1}; {x2} {y2}; {x3} {y3}"

    def sample_density(r_dx, r_dy, r_y1, r_yn):
        x, y = 1, sample_int(*r_y1)
        y_n = sample_int(*r_yn)
        xs, ys = [x], [y]

        while y > y_n:
            x += sample_int(*r_dx)
            y -= sample_int(*r_dy)
            y = max(y, y_n)
            xs.append(x)
            ys.append(y)

        return '; '.join(f"{x} {y}" for x, y in zip(xs, ys))

    num_days = sample_int(35,45)
    num_hours = num_days * 24
    
    CP = ControlParamSimple()
    CP.set_endDate(num_days=num_days)
    CP.set_value("comp1.heatingpipes.pipe1.@maxTemp", 60)
    CP.set_value("comp1.heatingpipes.pipe1.@minTemp", 0)
    CP.set_value("comp1.heatingpipes.pipe1.@radiationInfluence", "0 0")
    
    # ============== sample temp params ============== 
    CP.set_value("comp1.setpoints.temp.@heatingTemp", val_seq(sample_int, (5,30), num_hours))
    CP.set_value("comp1.setpoints.temp.@ventOffset", val_seq(sample_int, (0,5), num_hours))
    CP.set_value("comp1.setpoints.temp.@radiationInfluence", "0")
    CP.set_value("comp1.setpoints.temp.@PbandVent", val_seq(sample_line, [(0,5), (10,20), (20,25), (5,10)], num_hours))
    
    # ============== sample ventilation params ============== 
    CP.set_value("comp1.setpoints.ventilation.@startWnd", val_seq(sample_int, (0,50), num_hours))
    CP.set_value("comp1.setpoints.ventilation.@winLeeMin", 0)
    CP.set_value("comp1.setpoints.ventilation.@winLeeMax", 100)
    CP.set_value("comp1.setpoints.ventilation.@winWndMin", 0)
    CP.set_value("comp1.setpoints.ventilation.@winWndMax", 100)
    
    # ============== sample CO2 params ============== 
    CP.set_value("common.CO2dosing.@pureCO2cap", sample_int(100, 200))
    CP.set_value("comp1.setpoints.CO2.@setpoint", val_seq(sample_int, (400, 1200), num_hours))
    CP.set_value("comp1.setpoints.CO2.@setpIfLamps", val_seq(sample_int, (400, 1200), num_hours))
    CP.set_value("comp1.setpoints.CO2.@doseCapacity", val_seq(sample_doseCap, [(70,100), (40,70), (0,40)], num_hours))
    
    # ============== sample screen params ============== 
    sample_screen(CP, 1)
    sample_screen(CP, 2)
    
    # ============== sample illumination params ==============
    CP.set_value("comp1.illumination.lmp1.@enabled", True)
    CP.set_value("comp1.illumination.lmp1.@intensity", sample_int(50, 100))
    CP.set_value("comp1.illumination.lmp1.@endTime", 20)
    CP.set_hoursLight(val_seq(sample_int, (0, 20), num_days))
    CP.set_maxIglob(val_seq(sample_int, (200, 400), num_days))
    CP.set_maxPARsum(val_seq(sample_int, (10, 50), num_days))

    # ============== sample plant density ==============
    plant_density = sample_density(r_dx=(5,10), r_dy=(10,40), r_y1=(70,90), r_yn=(1,20))
    CP.set_value("crp_lettuce.Intkam.management.@plantDensity", plant_density)  # density must be in range of [1, 90]

    return CP


def make_CP():
    CP = ControlParamSimple()
    CP.set_endDate(num_days=42)
    CP.set_value("comp1.heatingpipes.pipe1.@maxTemp", 60)
    CP.set_value("comp1.heatingpipes.pipe1.@minTemp", 0)
    CP.set_value("comp1.heatingpipes.pipe1.@radiationInfluence", "0 0")
    
    # ============== sample temp params ============== 
    CP.set_value("comp1.setpoints.temp.@heatingTemp", {'01-01': {'r-1':5, 'r+1':20, 's-1':20, 's+1':5}})
    # CP.set_value("comp1.setpoints.temp.@ventOffset", {})
    # CP.set_value("comp1.setpoints.temp.@radiationInfluence", "0")
    # CP.set_value("comp1.setpoints.temp.@PbandVent", {})
    
    # ============== sample ventilation params ============== 
    # CP.set_value("comp1.setpoints.ventilation.@startWnd", {})
    # CP.set_value("comp1.setpoints.ventilation.@winLeeMin", 0)
    # CP.set_value("comp1.setpoints.ventilation.@winLeeMax", 100)
    # CP.set_value("comp1.setpoints.ventilation.@winWndMin", 0)
    # CP.set_value("comp1.setpoints.ventilation.@winWndMax", 100)
    
    # ============== sample CO2 params ============== 
    CP.set_value("common.CO2dosing.@pureCO2cap", 184)
    CP.set_value("comp1.setpoints.CO2.@setpoint", {'01-01': {'r-1':488, 'r+1':955, 's-1':955, 's+1':488}})
    CP.set_value("comp1.setpoints.CO2.@setpIfLamps", 990)
    # CP.set_value("comp1.setpoints.CO2.@doseCapacity", {})
    
    # ============== sample screen params ============== 
    # CP.set_value(f"comp1.screens.scr1.@enabled", True)
    # CP.set_value(f"comp1.screens.scr1.@material", 'scr_Transparent.par')
    # CP.set_value(f"comp1.screens.scr1.@ToutMax", {})
    # CP.set_value(f"comp1.screens.scr1.@closeBelow", {})
    # CP.set_value(f"comp1.screens.scr1.@closeAbove", {})
    # CP.set_value(f"comp1.screens.scr1.@lightPollutionPrevention", True)
    # CP.set_value(f"comp1.screens.scr2.@enabled", True)
    # CP.set_value(f"comp1.screens.scr2.@material", 'scr_Transparent.par')
    # CP.set_value(f"comp1.screens.scr2.@ToutMax", {})
    # CP.set_value(f"comp1.screens.scr2.@closeBelow", {})
    # CP.set_value(f"comp1.screens.scr2.@closeAbove", {})
    # CP.set_value(f"comp1.screens.scr2.@lightPollutionPrevention", True)
    
    # ============== sample illumination params ==============
    CP.set_value("comp1.illumination.lmp1.@enabled", True)
    CP.set_value("comp1.illumination.lmp1.@intensity", 64)
    CP.set_value("comp1.illumination.lmp1.@endTime", 20)
    CP.set_value("comp1.illumination.lmp1.@hoursLight", 6)
    CP.set_value("comp1.illumination.lmp1.@maxIglob", 267)

    # CP.set_hoursLight(6)
    # CP.set_maxIglob(267)
    # CP.set_maxPARsum({})

    # ============== sample plant density ==============
    CP.set_value("crp_lettuce.Intkam.management.@plantDensity", "1 90; 7 60; 14 40; 21 30; 28 20; 34 15")  # density must be in range of [1, 90]

    return CP


if __name__ == '__main__':
    import os
    import datetime
    import argparse
    from tqdm import tqdm
    from data import get_output, save_json_data

    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--simulator', type=str, default='A')
    parser.add_argument('-N', '--num-samples', type=int, required=True)
    args = parser.parse_args()

    today = datetime.date.today().isoformat()
    save_dir = f'../collect_data/data_sample=random_date={today}_sim={args.simulator}_number={args.num_samples}'
    control_dir = f'{save_dir}/controls'
    output_dir = f'{save_dir}/outputs'
    
    os.makedirs(control_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for _ in tqdm(range(args.num_samples)):
        CP = sample_CP_random()
        sample_name = f'{hex(hash(CP))}.json'
        CP.dump_json(control_dir, sample_name)
        
        output = get_output(CP.data, args.simulator)
        save_json_data(output, f'{output_dir}/{sample_name}')
