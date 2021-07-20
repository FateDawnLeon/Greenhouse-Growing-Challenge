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
    'C2BEST': {
        "num_days": 40,
        "heatingTemp_night": 9,
        "heatingTemp_day": 16,
        "CO2_pureCap": 159,
        "CO2_setpoint_night": 620,
        "CO2_setpoint_day": 1200,
        "CO2_setpoint_lamp": 1119,
        "light_intensity": 0,
        "light_hours": 10,
        "light_endTime": 20,
        "light_maxIglob": 299,
        "plantDensity": "1 80; 11 45; 19 25; 27 15",
    },
    'C3': {
        "num_days": tune.qrandint(lower=37, upper=40, q=1),
        "heatingTemp_night": tune.uniform(lower=9, upper=11),
        "heatingTemp_day": tune.uniform(lower=16, upper=21),
        "CO2_pureCap": tune.qrandint(lower=100, upper=300, q=10),
        "CO2_setpoint_night": tune.qrandint(lower=600, upper=700, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=1000, upper=1200, q=10),
        "CO2_setpoint_lamp": tune.qrandint(lower=1000, upper=1200, q=10),
        "light_intensity": tune.qrandint(lower=0, upper=500, q=10),
        "light_hours": tune.qrandint(lower=10, upper=15, q=1),
        "light_endTime": tune.qrandint(lower=17, upper=21, q=1),
        "light_maxIglob": tune.qrandint(lower=295, upper=305, q=5),
        "plantDensity": tune.choice(make_plant_density(37)),
    },
    'C3BEST': {
        "num_days": 37,
        "heatingTemp_night": 10.36368083752856,
        "heatingTemp_day": 19.31021249899327,
        "CO2_pureCap": 190,
        "CO2_setpoint_night": 670,
        "CO2_setpoint_day": 1050,
        "CO2_setpoint_lamp": 1170,
        "light_intensity": 50,
        "light_hours": 10,
        "light_endTime": 18,
        "light_maxIglob": 295,
        "plantDensity": "1 80; 11 45; 19 25; 27 15",
    },
    'C4': {
        "num_days": tune.qrandint(lower=35, upper=45, q=1),
        "heatingTemp_night": tune.uniform(lower=5, upper=15),
        "heatingTemp_day": tune.uniform(lower=15, upper=30),
        "CO2_pureCap": tune.qrandint(lower=100, upper=300, q=10),
        "CO2_setpoint_night": tune.qrandint(lower=600, upper=700, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=1000, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=0, upper=500, q=10),
        "light_hours": tune.uniform(lower=0, upper=18),
        "light_endTime": tune.uniform(lower=18, upper=20),
        "light_maxIglob": tune.qrandint(lower=100, upper=500, q=10),
        "plantDensity": tune.choice(make_plant_density(35)),
    },
    'C5': {
        "num_days": 37,
        "heatingTemp_night": tune.quniform(lower=10, upper=12, q=0.1),
        "heatingTemp_day": tune.quniform(lower=18, upper=20, q=0.1),
        "CO2_pureCap": 200,
        "CO2_setpoint_night": tune.qrandint(lower=650, upper=700, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=1000, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=0, upper=100, q=10),
        "light_hours": tune.quniform(lower=0, upper=18, q=0.1),
        "light_endTime": tune.quniform(lower=18, upper=20, q=0.1),
        "light_maxIglob": 300,
        "plantDensity": "1 80; 11 45; 19 25; 27 15",
    },
    'C5BEST': {
        "num_days": 37,
        "heatingTemp_night": 10.3,
        "heatingTemp_day": 19.4,
        "CO2_pureCap": 200,
        "CO2_setpoint_night": 690,
        "CO2_setpoint_day": 1110,
        "CO2_setpoint_lamp": 0,
        "light_intensity": 40,
        "light_hours": 7.8,
        "light_endTime": 19.3,
        "light_maxIglob": 300,
        "plantDensity": "1 80; 11 45; 19 25; 27 15",
    },
    'C6': {
        # best netprofit of this round: 5.821
        # best params: [40, 11.9, 19.6, 200, 650, 1180, 0, 0, 5.9, 19.8, 300, 9, 48, '1 90; 7 60; 14 40; 21 30; 28 20; 34 15']
        "num_days": tune.qrandint(lower=37, upper=40, q=1),
        "heatingTemp_night": tune.quniform(lower=10, upper=12, q=0.1),
        "heatingTemp_day": tune.quniform(lower=18, upper=20, q=0.1),
        "CO2_pureCap": 200,  # having minor effects on profit -> could be fixed
        "CO2_setpoint_night": tune.qrandint(lower=650, upper=700, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=1000, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,  # could be fixed to 0 so that only CO2_setpoint_day works
        "light_intensity": tune.qrandint(lower=0, upper=100, q=10),
        "light_hours": tune.quniform(lower=0, upper=18, q=0.1),
        "light_endTime": tune.quniform(lower=18, upper=20, q=0.1),
        "light_maxIglob": 300,  # having minor effects on profit -> could be fixed
        "scr1_ToutMax": tune.qrandint(lower=6, upper=10, q=1),  # 8
        "vent_startWnd": tune.qrandint(lower=46, upper=54, q=2),  # 50
        "plantDensity": tune.choice(make_plant_density(37)),
    },
    'SPA4BEST': {
        # best config: {'num_days': 38, 'heatingTemp_night': 10, 'heatingTemp_day': 26, 'CO2_pureCap': 182, 'CO2_setpoint_night': 694, 'CO2_setpoint_day': 1200, 'CO2_setpoint_lamp': 912, 'light_intensity': 0, 'light_hours': 12, 'light_endTime': 24, 'light_maxIglob': 300, 'scr1_ToutMax': 8, 'vent_startWnd': 50, 'plantDensity': '1 85; 9 50; 16 25; 26 20; 34 15'}
        # best netprofit of this round: 4.92
        "num_days": 38,
        "heatingTemp_night": 10,
        "heatingTemp_day": 26,
        "CO2_pureCap": 182,  # having minor effects on profit -> could be fixed
        "CO2_setpoint_night": 694,
        "CO2_setpoint_day": 1200,
        "CO2_setpoint_lamp": 912,  # could be fixed to 0 so that only CO2_setpoint_day works
        "light_intensity": 0,
        "light_hours": 12,
        "light_endTime": 24,
        "light_maxIglob": 300,  # having minor effects on profit -> could be fixed
        "scr1_ToutMax": 8, 
        "vent_startWnd": 50,
        "plantDensity": tune.choice(make_plant_density(38))
    },
    'SPC2BEST': {
        # best config: {'num_days': 40, 'heatingTemp_night': 9, 'heatingTemp_day': 16, 'CO2_pureCap': 159, 'CO2_setpoint_night': 620, 'CO2_setpoint_day': 1200, 'CO2_setpoint_lamp': 1119, 'light_intensity': 0, 'light_hours': 10, 'light_endTime': 20, 'light_maxIglob': 299, 'scr1_ToutMax': 8, 'vent_startWnd': 50, 'plantDensity': '1 80; 8 50; 17 35; 22 25; 29 15'}
        # best netprofit of this round: 6.725
        "num_days": 40,
        "heatingTemp_night": 9,
        "heatingTemp_day": 16,
        "CO2_pureCap": 159,
        "CO2_setpoint_night": 620,
        "CO2_setpoint_day": 1200,
        "CO2_setpoint_lamp": 1119,
        "light_intensity": 0,
        "light_hours": 10,
        "light_endTime": 20,
        "light_maxIglob": 299,
        "scr1_ToutMax": 8, 
        "vent_startWnd": 50,
        "plantDensity": tune.choice(make_plant_density(40))
    },
    'SPC2BEST1': {
        # generate 1000 plant density
        # best config: {'num_days': 40, 'heatingTemp_night': 9, 'heatingTemp_day': 16, 'CO2_pureCap': 159, 'CO2_setpoint_night': 620, 'CO2_setpoint_day': 1200, 'CO2_setpoint_lamp': 1119, 'light_intensity': 0, 'light_hours': 10, 'light_endTime': 20, 'light_maxIglob': 299, 'scr1_ToutMax': 8, 'vent_startWnd': 50, 'plantDensity': '1 90; 8 55; 16 35; 23 20; 33 15; 40 10'}
        # best netprofit of this round: 6.622
        "num_days": 40,
        "heatingTemp_night": 9,
        "heatingTemp_day": 16,
        "CO2_pureCap": 159,
        "CO2_setpoint_night": 620,
        "CO2_setpoint_day": 1200,
        "CO2_setpoint_lamp": 1119,
        "light_intensity": 0,
        "light_hours": 10,
        "light_endTime": 20,
        "light_maxIglob": 299,
        "scr1_ToutMax": 8, 
        "vent_startWnd": 50,
        "plantDensity": tune.choice(make_plant_density(40))
    },
    'SPC3BEST': {
        # best config: {'num_days': 37, 'heatingTemp_night': 10.36368083752856, 'heatingTemp_day': 19.31021249899327, 'CO2_pureCap': 190, 'CO2_setpoint_night': 670, 'CO2_setpoint_day': 1050, 'CO2_setpoint_lamp': 1170, 'light_intensity': 50, 'light_hours': 10, 'light_endTime': 18, 'light_maxIglob': 295, 'scr1_ToutMax': 8, 'vent_startWnd': 50, 'plantDensity': '1 80; 7 55; 16 35; 21 25; 28 15'}
        # best netprofit of this round: 6.592
        "num_days": 37,
        "heatingTemp_night": 10.36368083752856,
        "heatingTemp_day": 19.31021249899327,
        "CO2_pureCap": 190,
        "CO2_setpoint_night": 670,
        "CO2_setpoint_day": 1050,
        "CO2_setpoint_lamp": 1170,
        "light_intensity": 50,
        "light_hours": 10,
        "light_endTime": 18,
        "light_maxIglob": 295,
        "scr1_ToutMax": 8, 
        "vent_startWnd": 50,
        "plantDensity": tune.choice(make_plant_density(37))
    },
    'SPC6BEST': {
        # best config: {'num_days': 40, 'heatingTemp_night': 11.9, 'heatingTemp_day': 19.6, 'CO2_pureCap': 200, 'CO2_setpoint_night': 650, 'CO2_setpoint_day': 1180, 'CO2_setpoint_lamp': 0, 'light_intensity': 0, 'light_hours': 5.9, 'light_endTime': 19.8, 'light_maxIglob': 300, 'scr1_ToutMax': 9, 'vent_startWnd': 48, 'plantDensity': '1 85; 7 60; 17 40; 23 20; 30 15'}
        # best netprofit of this round: 3.983
        "num_days": 40,
        "heatingTemp_night": 11.9,
        "heatingTemp_day": 19.6,
        "CO2_pureCap": 200,  # having minor effects on profit -> could be fixed
        "CO2_setpoint_night": 650,
        "CO2_setpoint_day": 1180,
        "CO2_setpoint_lamp": 0,  # could be fixed to 0 so that only CO2_setpoint_day works
        "light_intensity": 0,
        "light_hours": 5.9,
        "light_endTime": 19.8,
        "light_maxIglob": 300,  # having minor effects on profit -> could be fixed
        "scr1_ToutMax": 9, 
        "vent_startWnd": 48,
        "plantDensity": tune.choice(make_plant_density(40))
    },
    'SPC6BEST1': {
        # generate 1000 plant density
        # best config: {'num_days': 40, 'heatingTemp_night': 11.9, 'heatingTemp_day': 19.6, 'CO2_pureCap': 200, 'CO2_setpoint_night': 650, 'CO2_setpoint_day': 1180, 'CO2_setpoint_lamp': 0, 'light_intensity': 0, 'light_hours': 5.9, 'light_endTime': 19.8, 'light_maxIglob': 300, 'scr1_ToutMax': 9, 'vent_startWnd': 48, 'plantDensity': '1 90; 7 60; 14 40; 21 30; 28 20; 34 15'}
        # best netprofit of this round: 5.821
        "num_days": 40,
        "heatingTemp_night": 11.9,
        "heatingTemp_day": 19.6,
        "CO2_pureCap": 200,  # having minor effects on profit -> could be fixed
        "CO2_setpoint_night": 650,
        "CO2_setpoint_day": 1180,
        "CO2_setpoint_lamp": 0,  # could be fixed to 0 so that only CO2_setpoint_day works
        "light_intensity": 0,
        "light_hours": 5.9,
        "light_endTime": 19.8,
        "light_maxIglob": 300,  # having minor effects on profit -> could be fixed
        "scr1_ToutMax": 9, 
        "vent_startWnd": 48,
        "plantDensity": tune.choice(make_plant_density(40))
    },
    'SPAUTOBEST': {
        # best config: {'num_days': 40, 'heatingTemp_night': 1, 'heatingTemp_day': 19, 'CO2_pureCap': 239, 'CO2_setpoint_night': 698, 'CO2_setpoint_day': 1198, 'CO2_setpoint_lamp': 1008, 'light_intensity': 87, 'light_hours': 3, 'light_endTime': 14, 'light_maxIglob': 295, 'scr1_ToutMax': 8, 'vent_startWnd': 50, 'plantDensity': '1 80; 10 55; 15 40; 20 25; 25 20; 31 15'}
        # best netprofit of this round: 7.227
        "num_days": 40,
        "heatingTemp_night": 1,
        "heatingTemp_day": 19,
        "CO2_pureCap": 239,
        "CO2_setpoint_night": 698,
        "CO2_setpoint_day": 1198,
        "CO2_setpoint_lamp": 1008,
        "light_intensity": 87,
        "light_hours": 3,
        "light_endTime": 14,
        "light_maxIglob": 295,
        "scr1_ToutMax": 8, 
        "vent_startWnd": 50,
        "plantDensity": tune.choice(make_plant_density(40))
    },
    'C7': {  #based on C3
        # best config: {'num_days': 37, 'heatingTemp_night': 10.830471190446758, 'heatingTemp_day': 19.70035032391595, 'CO2_pureCap': 300, 'CO2_setpoint_night': 670, 'CO2_setpoint_day': 1140, 'CO2_setpoint_lamp': 1200, 'light_intensity': 0, 'light_hours': 13, 'light_endTime': 18, 'light_maxIglob': 295, 'scr1_ToutMax': 6, 'vent_startWnd': 46, 'plantDensity': '1 80; 10 55; 15 40; 20 25; 25 20; 31 15'}
        # best netprofit of this round: 6.921
        "num_days": tune.qrandint(lower=37, upper=40, q=1),
        "heatingTemp_night": tune.uniform(lower=9, upper=11),
        "heatingTemp_day": tune.uniform(lower=16, upper=21),
        "CO2_pureCap": tune.qrandint(lower=100, upper=300, q=10),
        "CO2_setpoint_night": tune.qrandint(lower=600, upper=700, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=1000, upper=1200, q=10),
        "CO2_setpoint_lamp": tune.qrandint(lower=1000, upper=1200, q=10),
        "light_intensity": tune.qrandint(lower=0, upper=500, q=10),
        "light_hours": tune.qrandint(lower=10, upper=15, q=1),
        "light_endTime": tune.qrandint(lower=17, upper=21, q=1),
        "light_maxIglob": tune.qrandint(lower=295, upper=305, q=5),
        "scr1_ToutMax": tune.qrandint(lower=6, upper=10, q=1),  # 8
        "vent_startWnd": tune.qrandint(lower=46, upper=54, q=2),  # 50
        "plantDensity": '1 80; 10 55; 15 40; 20 25; 25 20; 31 15',
    },
    'C8': {  # UPDATE C7
        # best config: {'num_days': 37, 'heatingTemp_night': 10.434528809340254, 'heatingTemp_day': 17.450298410321942, 'CO2_pureCap': 270, 'CO2_setpoint_night': 640, 'CO2_setpoint_day': 1150, 'CO2_setpoint_lamp': 1170, 'light_intensity': 20, 'light_hours': 11, 'light_endTime': 19, 'light_maxIglob': 305, 'scr1_ToutMax': 7, 'vent_startWnd': 40, 'plantDensity': '1 80; 10 55; 15 40; 20 25; 25 20; 31 15'}
        # best netprofit of this round: 6.673
        "num_days": tune.qrandint(lower=35, upper=49, q=1),
        "heatingTemp_night": tune.uniform(lower=10, upper=10.5),
        "heatingTemp_day": tune.uniform(lower=17, upper=21),
        "CO2_pureCap": tune.qrandint(lower=250, upper=350, q=10),
        "CO2_setpoint_night": tune.qrandint(lower=600, upper=700, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=1100, upper=1200, q=10),
        "CO2_setpoint_lamp": tune.qrandint(lower=1150, upper=1200, q=10),
        "light_intensity": tune.qrandint(lower=0, upper=50, q=10),
        "light_hours": tune.qrandint(lower=10, upper=15, q=1),
        "light_endTime": tune.qrandint(lower=17, upper=20, q=1),
        "light_maxIglob": tune.qrandint(lower=285, upper=305, q=5),
        "scr1_ToutMax": tune.qrandint(lower=3, upper=8, q=1),  # 8
        "vent_startWnd": tune.qrandint(lower=40, upper=50, q=2),  # 50
        "plantDensity": '1 80; 10 55; 15 40; 20 25; 25 20; 31 15',
    },
    'C9': { # based on AUTO BEST
        # best config: {'num_days': 40, 'heatingTemp_night': 2.0, 'heatingTemp_day': 18.5, 'CO2_pureCap': 230, 'CO2_setpoint_night': 700, 'CO2_setpoint_day': 1110, 'CO2_setpoint_lamp': 1130, 'light_intensity': 30, 'light_hours': 1, 'light_endTime': 18.5, 'light_maxIglob': 285, 'scr1_ToutMax': 6, 'vent_startWnd': 48, 'plantDensity': '1 80; 10 55; 15 40; 20 25; 25 20; 31 15'}
        # best netprofit of this round: 7.702
        "num_days": tune.qrandint(lower=38, upper=42, q=1),
        "heatingTemp_night": tune.quniform(lower=1, upper=11, q=0.5),
        "heatingTemp_day": tune.quniform(lower=18, upper=20, q=0.5),
        "CO2_pureCap": tune.qrandint(lower=220, upper=260, q=10),  # having minor effects on profit -> could be fixed
        "CO2_setpoint_night": tune.qrandint(lower=680, upper=730, q=5),
        "CO2_setpoint_day": tune.qrandint(lower=1100, upper=1200, q=5),
        "CO2_setpoint_lamp": tune.qrandint(lower=990, upper=1200, q=10),  # could be fixed to 0 so that only CO2_setpoint_day works
        "light_intensity": tune.qrandint(lower=0, upper=100, q=10),
        "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=285, upper=305, q=5),  # having minor effects on profit -> could be fixed
        "scr1_ToutMax": tune.qrandint(lower=6, upper=10, q=1),  # 8
        "vent_startWnd": tune.qrandint(lower=46, upper=54, q=2),  # 50
        "plantDensity": '1 80; 10 55; 15 40; 20 25; 25 20; 31 15',
    },
    'C10': { # UPDATE C9
        # best config: {'num_days': 39, 'heatingTemp_night': 3.5, 'heatingTemp_day': 18.0, 'CO2_pureCap': 250, 'CO2_setpoint_night': 700, 'CO2_setpoint_day': 1180, 'CO2_setpoint_lamp': 1100, 'light_intensity': 50, 'light_hours': 5, 'light_endTime': 18.0, 'light_maxIglob': 295, 'scr1_ToutMax': 6, 'vent_startWnd': 50, 'plantDensity': '1 80; 10 55; 15 40; 20 25; 25 20; 31 15'}
        # best netprofit of this round: 7.816
        "num_days": tune.qrandint(lower=39, upper=41, q=1),
        "heatingTemp_night": tune.quniform(lower=1, upper=5, q=0.5),
        "heatingTemp_day": tune.quniform(lower=17, upper=19, q=0.5),
        "CO2_pureCap": tune.qrandint(lower=210, upper=250, q=10),  # 200; having minor effects on profit -> could be fixed
        "CO2_setpoint_night": tune.qrandint(lower=680, upper=720, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=1000, upper=1200, q=10),
        "CO2_setpoint_lamp": tune.qrandint(lower=1100, upper=1200, q=10),  # 0; could be fixed to 0 so that only CO2_setpoint_day works
        "light_intensity": tune.qrandint(lower=0, upper=50, q=10),
        "light_hours": tune.qrandint(lower=0, upper=5, q=1),
        "light_endTime": tune.quniform(lower=17.5, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=275, upper=295, q=5),  # 300; having minor effects on profit -> could be fixed
        "scr1_ToutMax": tune.qrandint(lower=3, upper=8, q=1),  # 8
        "vent_startWnd": tune.qrandint(lower=46, upper=52, q=2),  # 50
        "plantDensity": '1 80; 10 55; 15 40; 20 25; 25 20; 31 15',
    },
    'C11': { # UPDATE C10
        #     {
        #   "CO2_pureCap": 260,
        #   "CO2_setpoint_day": 1160,
        #   "CO2_setpoint_lamp": 1000,
        #   "CO2_setpoint_night": 700,
        #   "heatingTemp_day": 17.5,
        #   "heatingTemp_night": 4.5,
        #   "light_endTime": 18.5,
        #   "light_hours": 8,
        #   "light_intensity": 20,
        #   "light_maxIglob": 300,
        #   "num_days": 39,
        #   "plantDensity": "1 80; 10 55; 15 40; 20 25; 25 20; 31 15",
        #   "scr1_ToutMax": 5.0,
        #   "vent_startWnd": 51.0
        # }
        # best netprofit of this round: 7.892
        "num_days": tune.qrandint(lower=37, upper=41, q=1),
        "heatingTemp_night": tune.quniform(lower=1.5, upper=4.5, q=0.5),
        "heatingTemp_day": tune.quniform(lower=17, upper=19, q=0.5),
        "CO2_pureCap": tune.qrandint(lower=220, upper=280, q=10),  # 200; having minor effects on profit -> could be fixed
        "CO2_setpoint_night": tune.qrandint(lower=680, upper=720, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=1150, upper=1200, q=10),
        "CO2_setpoint_lamp": tune.qrandint(lower=1000, upper=1150, q=10),  # 0; could be fixed to 0 so that only CO2_setpoint_day works
        "light_intensity": tune.qrandint(lower=20, upper=80, q=10),
        "light_hours": tune.qrandint(lower=3, upper=10, q=1),
        "light_endTime": tune.quniform(lower=17.5, upper=19.5, q=0.5),
        "light_maxIglob": tune.qrandint(lower=285, upper=305, q=5),  # 300; having minor effects on profit -> could be fixed
        "scr1_ToutMax": tune.quniform(lower=5, upper=7, q=0.5), # 8
        "vent_startWnd": tune.quniform(lower=49, upper=51, q=0.5),  # 50
        "plantDensity": '1 80; 10 55; 15 40; 20 25; 25 20; 31 15',
    },
    'C12': { # UPDATE C11
    # best config: {'num_days': 39, 'heatingTemp_night': 3.5, 'heatingTemp_day': 18.0, 'CO2_pureCap': 260, 'CO2_setpoint_night': 710, 'CO2_setpoint_day': 1155, 'CO2_setpoint_lamp': 1010, 'light_intensity': 30, 'light_hours': 7, 'light_endTime': 18.6, 'light_maxIglob': 300, 'scr1_ToutMax': 5.5, 'vent_startWnd': 51.5, 'plantDensity': '1 80; 10 55; 15 40; 20 25; 25 20; 31 15'}
    # best netprofit of this round: 7.887
        "num_days": tune.qrandint(lower=38, upper=40, q=1),
        "heatingTemp_night": tune.quniform(lower=3, upper=6, q=0.5),
        "heatingTemp_day": tune.quniform(lower=16, upper=19, q=0.5),
        "CO2_pureCap": tune.qrandint(lower=250, upper=280, q=10),  # 200; having minor effects on profit -> could be fixed
        "CO2_setpoint_night": tune.qrandint(lower=690, upper=710, q=5),
        "CO2_setpoint_day": tune.qrandint(lower=1150, upper=1170, q=5),
        "CO2_setpoint_lamp": tune.qrandint(lower=950, upper=1050, q=10),  # 0; could be fixed to 0 so that only CO2_setpoint_day works
        "light_intensity": tune.qrandint(lower=0, upper=40, q=10),
        "light_hours": tune.qrandint(lower=7, upper=10, q=1),
        "light_endTime": tune.quniform(lower=18, upper=19, q=0.1),
        "light_maxIglob": tune.qrandint(lower=295, upper=305, q=2),  # 300; having minor effects on profit -> could be fixed
        "scr1_ToutMax": tune.quniform(lower=3, upper=7, q=0.5), # 8
        "vent_startWnd": tune.quniform(lower=49, upper=53, q=0.5),  # 50
        "plantDensity": '1 80; 10 55; 15 40; 20 25; 25 20; 31 15',
    },
    'C13': { # UPDATE C12
        # best config: {'num_days': 39, 'heatingTemp_night': 4.1000000000000005, 'heatingTemp_day': 17.7, 'CO2_pureCap': 262, 'CO2_setpoint_night': 725, 'CO2_setpoint_day': 1160, 'CO2_setpoint_lamp': 1030, 'light_intensity': 20, 'light_hours': 5, 'light_endTime': 19.0, 'light_maxIglob': 300, 'scr1_ToutMax': 4.6000000000000005, 'vent_startWnd': 52.5, 'plantDensity': '1 80; 10 55; 15 40; 20 25; 25 20; 31 15'}
        # best netprofit of this round: 7.94
        "num_days": 39,
        "heatingTemp_night": tune.quniform(lower=2.5, upper=4.5, q=0.1),
        "heatingTemp_day": tune.quniform(lower=17, upper=19, q=0.1),
        "CO2_pureCap": tune.qrandint(lower=255, upper=265, q=2),  # 200; having minor effects on profit -> could be fixed
        "CO2_setpoint_night": tune.qrandint(lower=670, upper=750, q=5),
        "CO2_setpoint_day": tune.qrandint(lower=1135, upper=1175, q=5),
        "CO2_setpoint_lamp": tune.qrandint(lower=990, upper=1030, q=5),  # 0; could be fixed to 0 so that only CO2_setpoint_day works
        "light_intensity": tune.qrandint(lower=0, upper=40, q=5),
        "light_hours": tune.qrandint(lower=5, upper=10, q=1),
        "light_endTime": tune.quniform(lower=18, upper=19, q=0.5),
        "light_maxIglob": 300, # 300; having minor effects on profit -> could be fixed
        "scr1_ToutMax": tune.quniform(lower=4, upper=6.4, q=0.2), # 8
        "vent_startWnd": tune.quniform(lower=50, upper=53, q=0.5),  # 50
        "plantDensity": '1 80; 10 55; 15 40; 20 25; 25 20; 31 15',
    },
     'C13BEST': { # BEST of C13
        # best netprofit of this round: 7.94
        "num_days": 39,
        "heatingTemp_night": 4.1,
        "heatingTemp_day": 17.7,
        "CO2_pureCap": 262,  # 200; having minor effects on profit -> could be fixed
        "CO2_setpoint_night": 725,
        "CO2_setpoint_day": 1160,
        "CO2_setpoint_lamp": 1030,  # 0; could be fixed to 0 so that only CO2_setpoint_day works
        "light_intensity":20,
        "light_hours": 5,
        "light_endTime": 19,
        "light_maxIglob": 300, # 300; having minor effects on profit -> could be fixed
        "scr1_ToutMax": 4.6, # 8
        "vent_startWnd": 52.5,  # 50
        "plantDensity": '1 80; 10 55; 15 40; 20 25; 25 20; 31 15',
    },
    'SPC13BEST': { # UPDATE C13
        # best config: {'num_days': 39, 'heatingTemp_night': 4.1, 'heatingTemp_day': 17.7, 'CO2_pureCap': 262, 'CO2_setpoint_night': 725, 'CO2_setpoint_day': 1160, 'CO2_setpoint_lamp': 1030, 'light_intensity': 20, 'light_hours': 5, 'light_endTime': 19, 'light_maxIglob': 300, 'scr1_ToutMax': 4.6, 'vent_startWnd': 53.5, 'plantDensity': '1 90; 6 55; 15 35; 23 20; 28 15'}
        # best netprofit of this round: 7.489
        "num_days": 39,
        "heatingTemp_night": 4.1,
        "heatingTemp_day":17.7,
        "CO2_pureCap": 262,  # 200; having minor effects on profit -> could be fixed
        "CO2_setpoint_night":725,
        "CO2_setpoint_day": 1160,
        "CO2_setpoint_lamp":1030,  # 0; could be fixed to 0 so that only CO2_setpoint_day works
        "light_intensity":20,
        "light_hours": 5,
        "light_endTime": 19,
        "light_maxIglob": 300, # 300; having minor effects on profit -> could be fixed
        "scr1_ToutMax": 4.6, # 8
        "vent_startWnd": 53.5,  # 50
        "plantDensity": tune.choice(make_plant_density(39)),
    },
    'C14': { # UPDATE C13
        # best config: {'num_days': 39, 'heatingTemp_night': 3.4000000000000004, 'heatingTemp_day': 18.400000000000002, 'CO2_pureCap': 262, 'CO2_setpoint_night': 705, 'CO2_setpoint_day': 1165, 'CO2_setpoint_lamp': 0, 'light_intensity': 20, 'light_hours': 8, 'light_endTime': 19.0, 'light_maxIglob': 300, 'scr1_ToutMax': 3.6, 'vent_startWnd': 54.0, 'plantDensity': '1 80; 10 55; 15 40; 20 25; 25 20; 31 15'}
        # best netprofit of this round: 7.873
        "num_days": 39,
        "heatingTemp_night": tune.quniform(lower=3, upper=4.6, q=0.2),
        "heatingTemp_day": tune.quniform(lower=17, upper=19, q=0.2),
        "CO2_pureCap": tune.qrandint(lower=259, upper=265, q=2),  # 200; having minor effects on profit -> could be fixed
        "CO2_setpoint_night": tune.qrandint(lower=700, upper=750, q=5),
        "CO2_setpoint_day": tune.qrandint(lower=1145, upper=1175, q=5),
        "CO2_setpoint_lamp": 0,  # 0; could be fixed to 0 so that only CO2_setpoint_day works
        "light_intensity": tune.qrandint(lower=0, upper=40, q=5),
        "light_hours": tune.qrandint(lower=3, upper=8, q=1),
        "light_endTime": tune.quniform(lower=18, upper=21, q=0.5),
        "light_maxIglob": 300, # 300; having minor effects on profit -> could be fixed
        "scr1_ToutMax": tune.quniform(lower=3.6, upper=5, q=0.2), # 8
        "vent_startWnd": tune.quniform(lower=51.5, upper=54, q=0.5),  # 50
        "plantDensity": '1 80; 10 55; 15 40; 20 25; 25 20; 31 15',
    },
    'C15': { # BASED ON C13BEST -> best netprofit: 7.94
        # best config: {'num_days': 37, 'heatingTemp_night': 3.3000000000000003, 'heatingTemp_day': 19.3, 'CO2_pureCap': 245, 'CO2_setpoint_night': 640, 'CO2_setpoint_day': 1140, 'CO2_setpoint_lamp': 0, 'light_intensity': 110, 'light_hours': 10, 'light_endTime': 18, 'light_maxIglob': 800, 'scr1_ToutMax': 5, 'vent_startWnd': 50, 'plantDensity': '1 80; 10 55; 15 40; 20 25; 25 20; 31 15'}
        # best netprofit of this round: 5.138
        "num_days": tune.qrandint(lower=35, upper=40, q=1),
        "heatingTemp_night": tune.quniform(lower=2, upper=5, q=0.1),
        "heatingTemp_day": tune.quniform(lower=17, upper=20, q=0.1),
        "CO2_pureCap": tune.qrandint(lower=200, upper=300, q=5),  # 200; having minor effects on profit -> could be fixed
        "CO2_setpoint_night": tune.qrandint(lower=600, upper=700, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=1100, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,  # tune.qrandint(lower=1000, upper=1200, q=10),  # 0; could be fixed to 0 so that only CO2_setpoint_day works
        "light_intensity": tune.qrandint(lower=100, upper=300, q=10),
        "light_hours": 10,  # tune.quniform(lower=10, upper=12, q=0.1),
        "light_endTime": 18,  # tune.quniform(lower=18, upper=19, q=0.1),
        "light_maxIglob": 800,  # 300; having minor effects on profit -> could be fixed
        "scr1_ToutMax": 5,  # tune.quniform(lower=4, upper=6.4, q=0.2), # 8
        "vent_startWnd": 50,  # tune.quniform(lower=50, upper=53, q=0.5),  # 50
        "plantDensity": tune.choice(make_plant_density(39)),
    },
    'C15BEST1': { # BASED ON C15 -> best netprofit: 5.138
        # best config: {'num_days': 40, 'heatingTemp_night': 3, 'heatingTemp_day': 20, 'CO2_pureCap': 270, 'CO2_setpoint_night': 400, 'CO2_setpoint_day': 1200, 'CO2_setpoint_lamp': 0, 'light_intensity': 0, 'light_hours': 9, 'light_endTime': 17, 'light_maxIglob': 800, 'scr1_ToutMax': 5, 'vent_startWnd': 50, 'plantDensity': '1 80; 10 55; 15 40; 20 25; 25 20; 31 15'}
        # best netprofit of this round: 7.796
        "num_days": 40,
        "heatingTemp_night": 3,
        "heatingTemp_day": 20,
        "CO2_pureCap": 270,
        "CO2_setpoint_night": 400,
        "CO2_setpoint_day": 1200,
        "CO2_setpoint_lamp": 0,
        "light_intensity": 0,
        "light_hours": 9,
        "light_endTime": 17,
        "light_maxIglob": 800,
        "scr1_ToutMax": 5,
        "vent_startWnd": 50,
        "plantDensity": '1 80; 10 55; 15 40; 20 25; 25 20; 31 15',
    },
    'C15BEST2': { # BASED ON C15 -> best netprofit: 5.138
        # best config: {'num_days': 40, 'heatingTemp_night': 3, 'heatingTemp_day': 20, 'CO2_pureCap': 270, 'CO2_setpoint_night': 400, 'CO2_setpoint_day': 1200, 'CO2_setpoint_lamp': 0, 'light_intensity': 0, 'light_hours': 9, 'light_endTime': 17, 'light_maxIglob': 800, 'scr1_ToutMax': 5, 'vent_startWnd': 50, 'plantDensity': '1 85; 7 50; 15 30; 25 20; 33 15'}
        # best netprofit of this round: 7.966
        "num_days": 40,
        "heatingTemp_night": 3,
        "heatingTemp_day": 20,
        "CO2_pureCap": 270,
        "CO2_setpoint_night": 400,
        "CO2_setpoint_day": 1200,
        "CO2_setpoint_lamp": 0,
        "light_intensity": 0,
        "light_hours": 9,
        "light_endTime": 17,
        "light_maxIglob": 800,
        "scr1_ToutMax": 5,
        "vent_startWnd": 50,
        "plantDensity": '1 85; 7 50; 15 30; 25 20; 33 15',
    },
    'D1': { # BASED ON C15TEST -> best netprofit: ???
        "num_days": 40,
        "heatingTemp_night": 3,
        "heatingTemp_day": 20,
        "CO2_pureCap": 270,
        "CO2_setpoint_night": 400,
        "CO2_setpoint_day": 1200,
        "CO2_setpoint_lamp": 1100,
        "light_intensity": 0,
        "light_hours": 9,
        "light_endTime": 17,
        "light_maxIglob": 800,
        "scr1_ToutMax": 5,
        "vent_startWnd": 50,
        "plantDensity": tune.choice(make_plant_density(35)),
    },
    'D1BEST': {
        "num_days": 40,
        "heatingTemp_night": 3,
        "heatingTemp_day": 20,
        "CO2_pureCap": 270,
        "CO2_setpoint_night": 400,
        "CO2_setpoint_day": 1200,
        "CO2_setpoint_lamp": 1100,
        "light_intensity": 0,
        "light_hours": 9,
        "light_endTime": 17,
        "light_maxIglob": 800,
        "scr1_ToutMax": 5,
        "vent_startWnd": 50,
        "plantDensity": "1 80; 9 45; 16 25; 23 20; 30 15",
    },
    'D2': { # BASED ON D1 -> best netprofit: ???
        "num_days": tune.qrandint(lower=35, upper=40, q=1),
        "heatingTemp_night": tune.quniform(lower=2, upper=10, q=0.5),
        "heatingTemp_day": tune.quniform(lower=15, upper=25, q=0.5),
        "CO2_pureCap": tune.qrandint(lower=260, upper=280, q=5),
        "CO2_setpoint_night": tune.qrandint(lower=400, upper=600, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=1100, upper=1200, q=10),
        "CO2_setpoint_lamp": tune.qrandint(lower=1100, upper=1200, q=10),
        "light_intensity": tune.qrandint(lower=0, upper=10, q=1),
        "light_hours": 9,
        "light_endTime": 17,
        "light_maxIglob": 800,
        "scr1_ToutMax": tune.quniform(lower=4, upper=6, q=0.2),
        "vent_startWnd": tune.quniform(lower=50, upper=55, q=0.5),
        "plantDensity": "1 80; 9 45; 16 25; 23 20; 30 15",  # TODO: need to adjust according to result from D1
    },
    'D2BEST': { # BASED ON D1 -> best netprofit: ???
        "num_days": 37,
        "heatingTemp_night": 5.5,
        "heatingTemp_day": 19,
        "CO2_pureCap": 275,
        "CO2_setpoint_night": 590,
        "CO2_setpoint_day": 1150,
        "CO2_setpoint_lamp": 1180,
        "light_intensity": 8,
        "light_hours": 9,
        "light_endTime": 17,
        "light_maxIglob": 800,
        "scr1_ToutMax": 5,
        "vent_startWnd": 51,
        "plantDensity": "1 80; 9 45; 16 25; 23 20; 30 15",  # TODO: need to adjust according to result from D1
    },
    'D3': { # BASED ON D2 -> best netprofit: ???
        "num_days": tune.qrandint(lower=36, upper=38, q=1),
        "heatingTemp_night": tune.quniform(lower=5, upper=7, q=0.1),
        "heatingTemp_day": tune.quniform(lower=18, upper=20, q=0.1),
        "CO2_pureCap": tune.qrandint(lower=260, upper=280, q=5),
        "CO2_setpoint_night": tune.qrandint(lower=400, upper=600, q=10),
        "CO2_setpoint_day": tune.qrandint(lower=1100, upper=1200, q=10),
        "CO2_setpoint_lamp": tune.qrandint(lower=1100, upper=1200, q=10),
        "light_intensity": tune.qrandint(lower=0, upper=10, q=1),
        "light_hours": 9,
        "light_endTime": 17,
        "light_maxIglob": 800,
        "scr1_ToutMax": tune.quniform(lower=3, upper=5, q=0.1),
        "vent_startWnd": tune.quniform(lower=48, upper=52, q=0.1),
        "plantDensity": "1 80; 9 45; 16 25; 23 20; 30 15",  # TODO: need to adjust according to result from D1
    },
    'D3BEST': { # BASED ON D2 -> best netprofit: ???
        "num_days": 37,
        "heatingTemp_night": 5.7,
        "heatingTemp_day": 19.3,
        "CO2_pureCap": 260,
        "CO2_setpoint_night": 510,
        "CO2_setpoint_day": 1140,
        "CO2_setpoint_lamp": 1190,
        "light_intensity": 1,
        "light_hours": 9,
        "light_endTime": 17,
        "light_maxIglob": 800,
        "scr1_ToutMax": 4.8,
        "vent_startWnd": 51.8,
        "plantDensity": "1 80; 9 45; 16 25; 23 20; 30 15",  # TODO: need to adjust according to result from D1
    },
    'D4': { # BASED ON D2 -> best netprofit: ???
        "num_days": 37,
        "heatingTemp_night": tune.quniform(lower=5.5, upper=6, q=0.1),
        "heatingTemp_day": tune.quniform(lower=19, upper=19.5, q=0.1),
        "CO2_pureCap": tune.qrandint(lower=250, upper=260, q=1),
        "CO2_setpoint_night": tune.qrandint(lower=500, upper=550, q=5),
        "CO2_setpoint_day": tune.qrandint(lower=1100, upper=1200, q=10),
        "CO2_setpoint_lamp": tune.qrandint(lower=1100, upper=1200, q=10),
        "light_intensity": tune.qrandint(lower=0, upper=5, q=1),
        "light_hours": tune.quniform(lower=8.5, upper=9.5, q=0.1),
        "light_endTime": 17,
        "light_maxIglob": 800,
        "scr1_ToutMax": tune.quniform(lower=4.5, upper=5, q=0.1),
        "vent_startWnd": tune.quniform(lower=50, upper=52, q=0.1),
        "plantDensity": "1 80; 9 45; 16 25; 23 20; 30 15",  # TODO: need to adjust according to result from D1
    },
    'D4BEST': {  # Netprofit=7.979
        'num_days': 37,
        'heatingTemp_night': 6.0,
        'heatingTemp_day': 19.5,
        'CO2_pureCap': 252,
        'CO2_setpoint_night': 545,
        'CO2_setpoint_day': 1130,
        'CO2_setpoint_lamp': 1100,
        'light_intensity': 3,
        'light_hours': 8.6,
        'light_endTime': 17,
        'light_maxIglob': 800,
        'scr1_ToutMax': 4.800000000000001,
        'vent_startWnd': 51.7,
        'plantDensity': '1 80; 9 45; 16 25; 23 20; 30 15'
        },
    'D5': {  # Netprofit=8.003
        'num_days': 37,
        'heatingTemp_night': 6.0,
        'heatingTemp_day': 19.5,
        'CO2_pureCap': 245,
        'CO2_setpoint_night': 545,
        'CO2_setpoint_day': 1130,
        'CO2_setpoint_lamp': 1100,
        'light_intensity': 3,
        'light_hours': 8.6,
        'light_endTime': 17,
        'light_maxIglob': 800,
        'scr1_ToutMax': 4.800000000000001,
        'vent_startWnd': 51.7,
        'plantDensity': '1 80; 9 45; 16 25; 23 20; 30 15'
        }
}


# best config: {'num_days': 37, 'heatingTemp_night': 10.36368083752856, 'heatingTemp_day': 19.31021249899327, 'CO2_pureCap': 190, 'CO2_setpoint_night': 670, 'CO2_setpoint_day': 1050, 'CO2_setpoint_lamp': 0, 'light_intensity': 50, 'light_hours': 10.0, 'light_endTime': 18.0, 'light_maxIglob': 295, 'plantDensity': '1 80; 11 45; 19 25; 27 15'}
# best netprofit: 6.713

# 2021-07-12 20:44:44,039 INFO tune.py:549 -- Total run time: 2504.97 seconds (2504.25 seconds for the tuning loop).
# best config: {'num_days': 37, 'heatingTemp_night': 10.3, 'heatingTemp_day': 19.400000000000002, 'CO2_pureCap': 200, 'CO2_setpoint_night': 690, 'CO2_setpoint_day': 1110, 'CO2_setpoint_lamp': 0, 'light_intensity': 40, 'light_hours': 7.800000000000001, 'light_endTime': 19.3, 'light_maxIglob': 300, 'plantDensity': '1 80; 11 45; 19 25; 27 15'}
# best netprofit: 6.87