from ray import tune
import numpy as np
import random


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


SPACES = {
    'G1': {
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
    'G2': {
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
        "plantDensity": '',
    },
}