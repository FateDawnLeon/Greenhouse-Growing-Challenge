from ray import tune
import numpy as np
import random


def pd_str2arr(max_days, pd_str):
    arr = np.zeros(max_days, dtype=np.int32)
    setpoints = [sp_str.split() for sp_str in pd_str.split('; ')]
    for day, val in setpoints:
        day = int(day)
        val = int(val)
        arr[day-1:] = val
    return arr


def compute_averageHeadPerM2(pd_arr):
    return len(pd_arr) / (1 / pd_arr).sum()


def compute_spacing_cost(max_days, pd_str):
    num_spacing_changes = len(pd_str.split('; ')) - 1
    fraction_of_year = max_days / 365
    return num_spacing_changes * 1.5 * fraction_of_year


def compute_max_return(max_days, pd_str):
    pd_arr = pd_str2arr(max_days, pd_str)
    avgHead = compute_averageHeadPerM2(pd_arr)
    max_product = 0.55 * avgHead
    plant_cost = avgHead * 0.12
    spacing_cost = compute_spacing_cost(max_days, pd_str)
    return max_product - plant_cost - spacing_cost


def make_plant_density(max_days):
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

    for i in range(1000):
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
            if days>max_days: break
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

    return control_densitys


def sample_pd(max_days, max_return_threshold=10):
    while True:
        setpoints = []
        min_density = random.choice([5, 10, 15])
        day, density = 1, random.choice([80, 85, 90])

        while day <= max_days and density >= min_density:
            setpoints.append(f'{day} {density}')
            day += random.choice([5,6,7,8,9,10])
            density -= random.choice([5,10,15,20,25,30,35])

        pd_str = '; '.join(setpoints)

        if compute_max_return(max_days, pd_str) > max_return_threshold:
            return pd_str


def optimize_pd(max_days, num_iters=1000):
    max_return = 0
    for _ in range(num_iters):
        setpoints = []
        min_density = random.choice([5, 10, 15])
        day, density = 1, random.choice([80, 85, 90])

        while day <= max_days and density >= min_density:
            setpoints.append(f'{day} {density}')
            day += random.choice([5,6,7,8,9,10])
            density -= random.choice([5,10,15,20,25,30,35])

        pd_str = '; '.join(setpoints)

        cur_return = compute_max_return(max_days, pd_str)
        if cur_return > max_return:
            max_return = cur_return
            best_pd = pd_str
    return best_pd, max_return


SPACES = {
    'G1': {  # netprofit=-5.003 and parameters={'duration': 35, 'heatingTemp_night': 7.5, 'heatingTemp_day': 21, 'CO2_pureCap': 280, 'CO2_setpoint_night': 480, 'CO2_setpoint_day': 1110, 'CO2_setpoint_lamp': 0, 'light_intensity': 20, 'light_hours': 8, 'light_endTime': 19.5, 'light_maxIglob': 260, 'scr1_ToutMax': 5, 'vent_startWnd': 55, 'plantDensity': '1 80; 6 45; 12 15; 20 5'}
        "duration": 35,
        "heatingTemp_night": 7.5,
        "heatingTemp_day": 21,
        "CO2_pureCap": 280,
        "CO2_setpoint_night": 480,
        "CO2_setpoint_day": 1110,
        "CO2_setpoint_lamp": 0,
        "light_intensity": 20,
        "light_hours": 8,
        "light_endTime": 19.5,
        "light_maxIglob": 260,
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        "plantDensity": tune.choice(make_plant_density(35)),
    },
    'G1BEST': {  # netprofit=-5.003 and parameters={'duration': 35, 'heatingTemp_night': 7.5, 'heatingTemp_day': 21, 'CO2_pureCap': 280, 'CO2_setpoint_night': 480, 'CO2_setpoint_day': 1110, 'CO2_setpoint_lamp': 0, 'light_intensity': 20, 'light_hours': 8, 'light_endTime': 19.5, 'light_maxIglob': 260, 'scr1_ToutMax': 5, 'vent_startWnd': 55, 'plantDensity': '1 80; 6 45; 12 15; 20 5'}
        "duration": 35,
        "heatingTemp_night": 7.5,
        "heatingTemp_day": 21,
        "CO2_pureCap": 280,
        "CO2_setpoint_night": 480,
        "CO2_setpoint_day": 1110,
        "CO2_setpoint_lamp": 0,
        "light_intensity": 20,
        "light_hours": 8,
        "light_endTime": 19.5,
        "light_maxIglob": 260,
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        "plantDensity": "1 80; 6 45; 12 15; 20 5",
    },
    'G2': { # netprofit=-4.374 and parameters={'duration': 36, 'heatingTemp_night': 3.5, 'heatingTemp_day': 16.0, 'CO2_pureCap': 265, 'CO2_setpoint_night': 600, 'CO2_setpoint_day': 1130, 'CO2_setpoint_lamp': 0, 'light_intensity': 0, 'light_hours': 11, 'light_endTime': 18.0, 'light_maxIglob': 200, 'scr1_ToutMax': 5.6000000000000005, 'vent_startWnd': 52.0, 'plantDensity': '1 80; 6 45; 12 15; 20 5'}
        "duration": tune.qrandint(lower=35, upper=45, q=1),
        "heatingTemp_night": tune.quniform(lower=2, upper=10, q=0.5),
        "heatingTemp_day": tune.quniform(lower=15, upper=25, q=0.5),
        "CO2_pureCap": tune.qrandint(lower=250, upper=280, q=5),
        "CO2_setpoint_night": tune.qrandint(lower=400, upper=600, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=1100, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=0, upper=100, q=5),
        "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=200, upper=500, q=10),
        "scr1_ToutMax": tune.quniform(lower=4, upper=6, q=0.2),
        "vent_startWnd": tune.quniform(lower=48, upper=57, q=1),
        "plantDensity": "1 80; 6 45; 12 15; 20 5",
    },
    'G2BEST': { # netprofit=-4.374 and parameters={'duration': 36, 'heatingTemp_night': 3.5, 'heatingTemp_day': 16.0, 'CO2_pureCap': 265, 'CO2_setpoint_night': 600, 'CO2_setpoint_day': 1130, 'CO2_setpoint_lamp': 0, 'light_intensity': 0, 'light_hours': 11, 'light_endTime': 18.0, 'light_maxIglob': 200, 'scr1_ToutMax': 5.6000000000000005, 'vent_startWnd': 52.0, 'plantDensity': '1 80; 6 45; 12 15; 20 5'}
        "duration": 36,
        "heatingTemp_night":  3.5,
        "heatingTemp_day": 16.0,
        "CO2_pureCap": 265,
        "CO2_setpoint_night": 600,
        "CO2_setpoint_day": 1130,
        "CO2_setpoint_lamp": 0,
        "light_intensity": 0,
        "light_hours": 11,
        "light_endTime": 18.0,
        "light_maxIglob": 200,
        "scr1_ToutMax": 5.6000000000000005,
        "vent_startWnd": 52,
        "plantDensity": "1 80; 6 45; 12 15; 20 5",
    },
    'G3': {  # netprofit=-4.307 and parameters={'duration': 37, 'heatingTemp_night': 3.5, 'heatingTemp_day': 16.0, 'CO2_pureCap': 260, 'CO2_setpoint_night': 300, 'CO2_setpoint_day': 410, 'CO2_setpoint_lamp': 0, 'light_intensity': 0, 'light_hours': 8, 'light_endTime': 18.0, 'light_maxIglob': 250, 'scr1_ToutMax': 5.0, 'vent_startWnd': 52.0, 'plantDensity': '1 80; 6 45; 12 15; 20 5'}
        "duration": tune.qrandint(lower=35, upper=45, q=1),
        "heatingTemp_night": tune.quniform(lower=2, upper=10, q=0.5),
        "heatingTemp_day": tune.quniform(lower=15, upper=25, q=0.5),
        "CO2_pureCap": tune.qrandint(lower=250, upper=280, q=5),
        "CO2_setpoint_night": tune.qrandint(lower=0, upper=400, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=400, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=0, upper=100, q=5),
        "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=200, upper=500, q=10),
        "scr1_ToutMax": tune.quniform(lower=4, upper=6, q=0.2),
        "vent_startWnd": tune.quniform(lower=48, upper=57, q=1),
        "plantDensity": "1 80; 6 45; 12 15; 20 5",
    },
    'G3BEST': {  # -4.281
        'duration': 36, 
        'heatingTemp_night': 3.5, 
        'heatingTemp_day': 16.5, 
        'CO2_pureCap': 265, 
        'CO2_setpoint_night': 190, 
        'CO2_setpoint_day': 460, 
        'CO2_setpoint_lamp': 0, 
        'light_intensity': 0, 
        'light_hours': 13, 
        'light_endTime': 18.0, 
        'light_maxIglob': 280, 
        'scr1_ToutMax': 5.4, 
        'vent_startWnd': 48.0, 
        'plantDensity': '1 80; 6 45; 12 15; 20 5'
    },
    'G4': { 
        "duration": tune.qrandint(lower=35, upper=60, q=2),
        "heatingTemp_night": tune.quniform(lower=2, upper=10, q=0.5),
        "heatingTemp_day": tune.quniform(lower=15, upper=25, q=0.5),
        "CO2_pureCap": tune.qrandint(lower=250, upper=280, q=5),
        "CO2_setpoint_night": tune.qrandint(lower=0, upper=400, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=400, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=0, upper=500, q=20),
        "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=200, upper=500, q=10),
        "scr1_ToutMax": tune.quniform(lower=4, upper=6, q=0.2),
        "vent_startWnd": tune.quniform(lower=48, upper=57, q=1),
        "plantDensity": "1 80; 6 45; 12 15; 20 5",
    },
    'G5': { 
        "duration": tune.qrandint(lower=38, upper=42, q=1),
        "heatingTemp_night": tune.quniform(lower=2, upper=10, q=0.5),
        "heatingTemp_day": tune.quniform(lower=15, upper=25, q=0.5),
        "CO2_pureCap": tune.qrandint(lower=250, upper=280, q=5),
        "CO2_setpoint_night": tune.qrandint(lower=0, upper=400, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=400, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=100, upper=500, q=20),
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=200, upper=500, q=10),
        "scr1_ToutMax": tune.quniform(lower=4, upper=6, q=0.2),
        "vent_startWnd": tune.quniform(lower=48, upper=57, q=1),
        "plantDensity": "1 80; 6 45; 12 15; 20 5",
    },
    'G6': { 
        "duration": tune.qrandint(lower=38, upper=42, q=1),
        "heatingTemp_night": tune.quniform(lower=2, upper=10, q=0.5),
        "heatingTemp_day": tune.quniform(lower=15, upper=25, q=0.5),
        "CO2_pureCap": tune.qrandint(lower=250, upper=280, q=5),
        "CO2_setpoint_night": tune.qrandint(lower=0, upper=400, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=400, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=0, upper=100, q=20),
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": 800,
        "scr1_ToutMax": tune.quniform(lower=4, upper=6, q=0.2),
        "vent_startWnd": tune.quniform(lower=48, upper=57, q=1),
        "plantDensity": "1 80; 6 45; 12 15; 20 5",
    },
    'G7': { 
        "duration": tune.qrandint(lower=38, upper=50, q=1),
        "heatingTemp_night": 5,
        "heatingTemp_day": 20,
        "CO2_pureCap": 300,
        "CO2_setpoint_night": 400,
        "CO2_setpoint_day": 1200,
        "CO2_setpoint_lamp": 0,
        "light_intensity": 0,
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": 800,
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        "plantDensity": "1 80; 6 45; 12 15; 20 5",
    },  
    'G8': { 
        "duration": 40,
        "heatingTemp_night": 5,
        "heatingTemp_day": 20,
        "CO2_pureCap": 300,
        "CO2_setpoint_night": 400,
        "CO2_setpoint_day": 1200,
        "CO2_setpoint_lamp": 0,
        "light_intensity": 0,
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": 800,
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        "plantDensity": tune.choice(make_plant_density(40)),
    },
    'G9': { 
        "duration": 40,
        "heatingTemp_night": 10,
        "heatingTemp_day": 25,
        "CO2_pureCap": 300,
        "CO2_setpoint_night": 400,
        "CO2_setpoint_day": 1200,
        "CO2_setpoint_lamp": 0,
        "light_intensity": 100,
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": 800,
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        "plantDensity":  '1 90; 8 65; 18 45; 26 25; 35 15',
    },
    'G10': { 
        "duration": 40,
        "heatingTemp_night": 10,
        "heatingTemp_day": 25,
        "CO2_pureCap": 300,
        "CO2_setpoint_night": 400,
        "CO2_setpoint_day": 1200,
        "CO2_setpoint_lamp": 0,
        "light_intensity": 100,
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": 800,
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        "plantDensity":  '1 90; 26 80; 35 60; 40 40',
    },
    'G11': { 
        "duration": 40,
        "heatingTemp_night": 10,
        "heatingTemp_day": 25,
        "CO2_pureCap": 300,
        "CO2_setpoint_night": 400,
        "CO2_setpoint_day": 1200,
        "CO2_setpoint_lamp": 0,
        "light_intensity": 130,
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": 800,
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        # "plantDensity": '1 80; 6 50; 12 25; 18 5',
        "plantDensity": tune.sample_from(lambda spec: sample_pd(spec.config.duration)),
    },

}