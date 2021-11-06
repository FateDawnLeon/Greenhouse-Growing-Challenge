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

    control_densitys = set([
        "1 80; 11 45; 19 25; 27 15",
        "1 90; 7 60; 14 40; 21 30; 28 20; 34 15",
        "1 80; 9 50; 14 25; 20 20; 27 15",
        "1 80; 12 45; 20 25; 27 20; 35 10",  # from email
        "1 80; 10 40; 20 30; 25 20; 30 10",  # from control sample
        "1 80; 10 55; 15 40; 20 25; 25 20; 31 15",  # from C15TEST
        "1 85; 7 50; 15 30; 25 20; 33 15",  # from D1 test on sim C
    ])

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
            if days > max_days:
                break
            density = density - change_density
            if density < end_density:
                break
            skip_days.append(skip_day)
            change_densitys.append(change_density)
        change_densitys.sort(reverse=True)

        days = 1
        density = start_density
        control_density = f'{days} {density}'
        for i in range(len(skip_days)):
            days += skip_days[i]
            density = density - change_densitys[i]
            control_density = f'{control_density}; {days} {density}'

        if density in end_density_range:
            control_densitys.add(control_density)

    return list(control_densitys)


def sample_pd(max_days, max_return_threshold=10):
    while True:
        setpoints = []
        min_density = random.choice([5, 10, 15])
        day, density = 1, random.choice([80, 85, 90])

        while day <= max_days and density >= min_density:
            setpoints.append(f'{day} {density}')
            day += random.choice([5, 6, 7, 8, 9, 10])
            density -= random.choice([5, 10, 15, 20, 25, 30, 35])

        pd_str = '; '.join(setpoints)

        # print ('pd_str:', pd_str)

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
def sample_list_pd(max_days, iter):
    control_densitys = set()
    for i in range(iter):
        control_densitys.add(sample_pd(max_days, max_return_threshold=10))
    print(list(control_densitys))
    return list(control_densitys)


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
    'G2': {  # netprofit=-4.374 and parameters={'duration': 36, 'heatingTemp_night': 3.5, 'heatingTemp_day': 16.0, 'CO2_pureCap': 265, 'CO2_setpoint_night': 600, 'CO2_setpoint_day': 1130, 'CO2_setpoint_lamp': 0, 'light_intensity': 0, 'light_hours': 11, 'light_endTime': 18.0, 'light_maxIglob': 200, 'scr1_ToutMax': 5.6000000000000005, 'vent_startWnd': 52.0, 'plantDensity': '1 80; 6 45; 12 15; 20 5'}
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
    'G2BEST': {  # netprofit=-4.374 and parameters={'duration': 36, 'heatingTemp_night': 3.5, 'heatingTemp_day': 16.0, 'CO2_pureCap': 265, 'CO2_setpoint_night': 600, 'CO2_setpoint_day': 1130, 'CO2_setpoint_lamp': 0, 'light_intensity': 0, 'light_hours': 11, 'light_endTime': 18.0, 'light_maxIglob': 200, 'scr1_ToutMax': 5.6000000000000005, 'vent_startWnd': 52.0, 'plantDensity': '1 80; 6 45; 12 15; 20 5'}
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
    'G12': {  # fractiongroundcover = 0.95
        "duration": 33,
        "heatingTemp_night": 10,  # 7 ; 10-15
        "heatingTemp_day": 22,  # 20, 22(23)
        "CO2_pureCap": 300,
        "CO2_setpoint_night": 0,  # 0
        "CO2_setpoint_day": 1000,  # 800; 1000
        "CO2_setpoint_lamp": 0,
        "light_intensity": 160,  # 200
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1), # 15-20
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": 100,
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        # "plantDensity": '1 80; 6 50; 12 25; 18 5',
        "plantDensity": "1 78; 7 38; 13 25; 19 20; 26 15"
    },
    'G13': {
        "duration": 33,
        "heatingTemp_night": tune.quniform(lower=7, upper=13, q=1),
        "heatingTemp_day": tune.quniform(lower=18, upper=25, q=1),
        "CO2_pureCap": 300,
        "CO2_setpoint_night": tune.qrandint(lower=0, upper=50, q=5),
        "CO2_setpoint_day": tune.qrandint(lower=800, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=50, upper=200, q=5),
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=50, upper=200, q=5),
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        "plantDensity": "1 78; 7 38; 13 25; 19 20; 26 15"
    },
    #     Netprofit=-0.1, Config={'duration': 33, 'heatingTemp_night': 9.0, 'heatingTemp_day': 21.0, 'CO2_pureCap': 300, 'CO2_setpoint_night': 0, 'CO2_setpoint_day': 930, 'CO2_setpoint_lamp': 0, 'light_intensity': 180, 'light_maxIglob': 90, 'scr1_ToutMax': 5, 'vent_startWnd': 55, 'plantDensity': '1 78; 7 38; 13 25; 19 20; 26 15'}
    # (ImplicitFunc pid=2793538) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.09, 'AverageHeadm2': 24.3, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.0}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.537, 'comp1.Lmp1.Costs': 0.64, 'comp1.Scr1.Costs': 0.113, 'comp1.ConCO2.Costs': 0.407, 'spacingSystem': 0.542, 'plants': 2.916}, 'total': 6.155}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.826, 'comp1.Lmp1.ElecUse': 4.338, 'CO2': 0.287}, 'total': 5.451}, 'gains': {'objects': {'product': 11.506}, 'total': 11.506}, 'balance': -0.1}
    # (ImplicitFunc pid=2793538) comp1.Plant.headFW 244.7093955135521
    # (ImplicitFunc pid=2793538) comp1.Plant.fractionGroundCover 0.9741463343867571
    # (ImplicitFunc pid=2793538) comp1.Plant.shootDryMatterContent 0.04935390213878333
    # (ImplicitFunc pid=2793538) comp1.Plant.qualityLoss 0.0

    # (ImplicitFunc pid=2793070) Netprofit=0.109, Config={'duration': 33, 'heatingTemp_night': 10.0, 'heatingTemp_day': 23.0, 'CO2_pureCap': 300, 'CO2_setpoint_night': 20, 'CO2_setpoint_day': 850, 'CO2_setpoint_lamp': 0, 'light_intensity': 150, 'light_maxIglob': 180, 'scr1_ToutMax': 5, 'vent_startWnd': 55, 'plantDensity': '1 78; 7 38; 13 25; 19 20; 26 15'}
    # (ImplicitFunc pid=2793070) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.09, 'AverageHeadm2': 24.3, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.281}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.537, 'comp1.Lmp1.Costs': 0.533, 'comp1.Scr1.Costs': 0.113, 'comp1.ConCO2.Costs': 0.407, 'spacingSystem': 0.542, 'plants': 2.916}, 'total': 6.048}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.955, 'comp1.Lmp1.ElecUse': 4.441, 'CO2': 0.286}, 'total': 5.682}, 'gains': {'objects': {'product': 11.839}, 'total': 11.839}, 'balance': 0.109}
    # (ImplicitFunc pid=2793070) comp1.Plant.headFW 249.76290896737999
    # (ImplicitFunc pid=2793070) comp1.Plant.fractionGroundCover 0.9481051867957793
    # (ImplicitFunc pid=2793070) comp1.Plant.shootDryMatterContent 0.048624113535897996
    # (ImplicitFunc pid=2793070) comp1.Plant.qualityLoss 2.319086170505219
    'G13BEST': {  # Netprofit=0.109
        'duration': 33,
        'heatingTemp_night': 10.0,
        'heatingTemp_day': 23.0,
        'CO2_pureCap': 300,
        'CO2_setpoint_night': 20,
        'CO2_setpoint_day': 850,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 150,
        'light_maxIglob': 180,
        'scr1_ToutMax': 5,
        'vent_startWnd': 55,
        'plantDensity': '1 78; 7 38; 13 25; 19 20; 26 15'
    },
    'G14': {
        "duration": 34,
        "heatingTemp_night": tune.quniform(lower=7, upper=13, q=1),
        "heatingTemp_day": tune.quniform(lower=18, upper=25, q=1),
        "CO2_pureCap": 300,
        "CO2_setpoint_night": tune.qrandint(lower=0, upper=50, q=5),
        "CO2_setpoint_day": tune.qrandint(lower=800, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=50, upper=200, q=5),
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=50, upper=200, q=5),
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        "plantDensity": "1 70; 8 37; 13 28; 19 20; 27 15"
    },
    #     Netprofit=0.401, Config={'duration': 34, 'heatingTemp_night': 12.0, 'heatingTemp_day': 21.0, 'CO2_pureCap': 300, 'CO2_setpoint_night': 50, 'CO2_setpoint_day': 830, 'CO2_setpoint_lamp': 0, 'light_intensity': 170, 'light_maxIglob': 80, 'scr1_ToutMax': 5, 'vent_startWnd': 55, 'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'}
    # (ImplicitFunc pid=2833464) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.093, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.182}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.584, 'comp1.Lmp1.Costs': 0.623, 'comp1.Scr1.Costs': 0.116, 'comp1.ConCO2.Costs': 0.419, 'spacingSystem': 0.559, 'plants': 2.951}, 'total': 6.251}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 1.001, 'comp1.Lmp1.ElecUse': 4.031, 'CO2': 0.253}, 'total': 5.285}, 'gains': {'objects': {'product': 11.937}, 'total': 11.937}, 'balance': 0.401}
    # (ImplicitFunc pid=2833464) comp1.Plant.headFW 248.57224112446283
    # (ImplicitFunc pid=2833464) comp1.Plant.fractionGroundCover 0.9695689642751161
    # (ImplicitFunc pid=2833464) comp1.Plant.shootDryMatterContent 0.04886556766416179
    # (ImplicitFunc pid=2833464) comp1.Plant.qualityLoss 2.0
    'G14BEST': {
        'duration': 34,
        'heatingTemp_night': 12.0,
        'heatingTemp_day': 21.0,
        'CO2_pureCap': 300,
        'CO2_setpoint_night': 50,
        'CO2_setpoint_day': 830,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 170,
        'light_maxIglob': 80,
        'scr1_ToutMax': 5,
        'vent_startWnd': 55,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'
    },
    'G15': {
        'duration': 34,
        'heatingTemp_night': 12.0,
        'heatingTemp_day': 21.0,
        'CO2_pureCap': 300,
        'CO2_setpoint_night': 50,
        'CO2_setpoint_day': 830,
        'CO2_setpoint_lamp': 0,
        "light_intensity": tune.qrandint(lower=50, upper=200, q=5),
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=50, upper=200, q=5),
        'scr1_ToutMax': 5,
        'vent_startWnd': 55,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'
    },
    #     Netprofit=0.719, Config={'duration': 34, 'heatingTemp_night': 12.0, 'heatingTemp_day': 21.0, 'CO2_pureCap': 300, 'CO2_setpoint_night': 50, 'CO2_setpoint_day': 830, 'CO2_setpoint_lamp': 0, 'light_intensity': 165, 'light_maxIglob': 100, 'scr1_ToutMax': 5, 'vent_startWnd': 55, 'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'}
    # (ImplicitFunc pid=2868488) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.093, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.0}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.584, 'comp1.Lmp1.Costs': 0.604, 'comp1.Scr1.Costs': 0.116, 'comp1.ConCO2.Costs': 0.419, 'spacingSystem': 0.559, 'plants': 2.951}, 'total': 6.233}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.979, 'comp1.Lmp1.ElecUse': 4.017, 'CO2': 0.251}, 'total': 5.248}, 'gains': {'objects': {'product': 12.2}, 'total': 12.2}, 'balance': 0.719}
    # (ImplicitFunc pid=2868488) comp1.Plant.headFW 249.23234639415764
    # (ImplicitFunc pid=2868488) comp1.Plant.fractionGroundCover 0.9743343441461859
    # (ImplicitFunc pid=2868488) comp1.Plant.shootDryMatterContent 0.048678289602737616
    # (ImplicitFunc pid=2868488) comp1.Plant.qualityLoss 0.0
    'G15BEST': {
        'duration': 34,
        'heatingTemp_night': 12.0,
        'heatingTemp_day': 21.0,
        'CO2_pureCap': 300,
        'CO2_setpoint_night': 50,
        'CO2_setpoint_day': 830,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 165,
        'light_maxIglob': 100,
        'scr1_ToutMax': 5,
        'vent_startWnd': 55,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'
    },
    'G15BESTCAP': {'duration': 34,
                   'heatingTemp_night': 12.0,
                   'heatingTemp_day': 21.0,
                   'CO2_pureCap': 109,
                   'CO2_setpoint_night': 50,
                   'CO2_setpoint_day': 830,
                   'CO2_setpoint_lamp': 0,
                   'light_intensity': 165,
                   'light_maxIglob': 100,
                   'scr1_ToutMax': 5,
                   'vent_startWnd': 55,
                   'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'
                   },
    # Netprofit=0.959, Config={'duration': 34, 'heatingTemp_night': 12.0, 'heatingTemp_day': 21.0, 'CO2_pureCap': 109, 'CO2_setpoint_night': 50, 'CO2_setpoint_day': 830, 'CO2_setpoint_lamp': 0, 'light_intensity': 165, 'light_maxIglob': 100, 'scr1_ToutMax': 5, 'vent_startWnd': 55, 'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'}
    # (ImplicitFunc pid=3081311) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.093, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.0}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.584, 'comp1.Lmp1.Costs': 0.604, 'comp1.Scr1.Costs': 0.116, 'comp1.ConCO2.Costs': 0.152, 'spacingSystem': 0.559, 'plants': 2.951}, 'total': 5.966}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.982, 'comp1.Lmp1.ElecUse': 4.017, 'CO2': 0.239}, 'total': 5.238}, 'gains': {'objects': {'product': 12.163}, 'total': 12.163}, 'balance': 0.959}
    # (ImplicitFunc pid=3081311) comp1.Plant.headFW 248.93372660781472
    # (ImplicitFunc pid=3081311) comp1.Plant.fractionGroundCover 0.9739553433783031
    # (ImplicitFunc pid=3081311) comp1.Plant.shootDryMatterContent 0.048658339399427646
    # (ImplicitFunc pid=3081311) comp1.Plant.qualityLoss 0.0
    'G15LAMP': {
        'duration': 34,
        'heatingTemp_night': 12.0,
        'heatingTemp_day': 21.0,
        'CO2_pureCap': 300,
        'CO2_setpoint_night': 50,
        'CO2_setpoint_day': 830,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 165,
        'light_maxIglob': 100,
        "lamp_type": "lmp_LED32.par",  # lmp_SON-T1000W.par, lmp_LED29.par, lmp_LED32.par
        'scr1_ToutMax': 5,
        'vent_startWnd': 55,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'
    },
    # 'G16': {
    #         "duration": 41,
    #         "heatingTemp_night": tune.quniform(lower=7, upper=13, q=1),
    #         "heatingTemp_day": tune.quniform(lower=18, upper=25, q=1),
    #         "CO2_pureCap": 300,
    #         "CO2_setpoint_night": tune.qrandint(lower=0, upper=50, q=5),
    #         "CO2_setpoint_day": tune.qrandint(lower=800, upper=1200, q=10),
    #         "CO2_setpoint_lamp": 0,
    #         "light_intensity": tune.qrandint(lower=50, upper=200, q=5),
    #         # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
    #         # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
    #         "light_maxIglob": tune.qrandint(lower=50, upper=200, q=5),
    #         "scr1_ToutMax": 5,
    #         "vent_startWnd": 55,
    #         "plantDensity": "1 78; 8 38; 17 25; 26 20; 31 15"
    #     },
    'G16': {  # 0.175;  0.179: 'light_intensity': 228, 'light_maxIglob': 396
        'duration': 25,
        'heatingTemp_night': 9.0,
        'heatingTemp_day': 24.0,
        'CO2_pureCap': 300,
        'CO2_setpoint_night': 25,
        'CO2_setpoint_day': 1080,
        'CO2_setpoint_lamp': 0,
        "light_intensity": 230,
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": 380,
        'scr1_ToutMax': 5,
        'vent_startWnd': 55,
        'plantDensity': '1 72; 6 34; 12 21; 19 15'
    },
    'G17': {
        "duration": 40,
        "heatingTemp_night": tune.quniform(lower=7, upper=13, q=1),
        "heatingTemp_day": tune.quniform(lower=18, upper=25, q=1),
        "CO2_pureCap": 300,
        "CO2_setpoint_night": tune.qrandint(lower=0, upper=50, q=5),
        "CO2_setpoint_day": tune.qrandint(lower=800, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=50, upper=200, q=5),
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=50, upper=200, q=5),
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        "plantDensity": "1 74; 5 47; 14 35; 18 26; 24 19; 31 15"
    },
    #     Netprofit=0.293, Config={'duration': 40, 'heatingTemp_night': 10.0, 'heatingTemp_day': 20.0, 'CO2_pureCap': 300, 'CO2_setpoint_night': 0, 'CO2_setpoint_day': 1200, 'CO2_setpoint_lamp': 0, 'light_intensity': 110, 'light_maxIglob': 105, 'scr1_ToutMax': 5, 'vent_startWnd': 55, 'plantDensity': '1 74; 5 47; 14 35; 18 26; 24 19; 31 15'}
    # (ImplicitFunc pid=2972212) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.11, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.613}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.863, 'comp1.Lmp1.Costs': 0.474, 'comp1.Scr1.Costs': 0.137, 'comp1.ConCO2.Costs': 0.493, 'spacingSystem': 0.822, 'plants': 2.953}, 'total': 6.742}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 1.197, 'comp1.Lmp1.ElecUse': 3.038, 'CO2': 0.368}, 'total': 4.603}, 'gains': {'objects': {'product': 11.638}, 'total': 11.638}, 'balance': 0.293}
    # (ImplicitFunc pid=2972212) comp1.Plant.headFW 250.42307418298802
    # (ImplicitFunc pid=2972212) comp1.Plant.fractionGroundCover 0.9906077751220242
    # (ImplicitFunc pid=2972212) comp1.Plant.shootDryMatterContent 0.047918944056315256
    # (ImplicitFunc pid=2972212) comp1.Plant.qualityLoss 6.5
    'G17BEST': {
        'duration': 40,
        'heatingTemp_night': 10.0,
        'heatingTemp_day': 20.0,
        'CO2_pureCap': 300,
        'CO2_setpoint_night': 0,
        'CO2_setpoint_day': 1200,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 110,
        'light_maxIglob': 105,
        'scr1_ToutMax': 5,
        'vent_startWnd': 55,
        'plantDensity': '1 74; 5 47; 14 35; 18 26; 24 19; 31 15'
    },
    'G17BESTCUT': {
        'duration': 35,
        'heatingTemp_night': 10.0,
        'heatingTemp_day': 20.0,
        'CO2_pureCap': 300,
        'CO2_setpoint_night': 0,
        'CO2_setpoint_day': 1200,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 110,
        'light_maxIglob': 105,
        'scr1_ToutMax': 5,
        'vent_startWnd': 55,
        'plantDensity': '1 74; 5 47; 14 35; 18 26; 24 19; 31 15'
    },
    'G18': {
        "duration": 36,
        "heatingTemp_night": tune.quniform(lower=7, upper=13, q=1),
        "heatingTemp_day": tune.quniform(lower=18, upper=25, q=1),
        "CO2_pureCap": 300,
        "CO2_setpoint_night": tune.qrandint(lower=0, upper=50, q=5),
        "CO2_setpoint_day": tune.qrandint(lower=800, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=50, upper=200, q=5),
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=50, upper=200, q=5),
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        "plantDensity": "1 78; 4 75; 8 35; 15 26; 22 20; 28 15"
    },
    #     Netprofit=-0.039, Config={'duration': 36, 'heatingTemp_night': 10.0, 'heatingTemp_day': 21.0, 'CO2_pureCap': 300, 'CO2_setpoint_night': 45, 'CO2_setpoint_day': 1030, 'CO2_setpoint_lamp': 0, 'light_intensity': 140, 'light_maxIglob': 130, 'scr1_ToutMax': 5, 'vent_startWnd': 55, 'plantDensity': '1 78; 4 75; 8 35; 15 26; 22 20; 28 15'}
    # (ImplicitFunc pid=2978607) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.099, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.423}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.677, 'comp1.Lmp1.Costs': 0.543, 'comp1.Scr1.Costs': 0.123, 'comp1.ConCO2.Costs': 0.444, 'spacingSystem': 0.74, 'plants': 2.957}, 'total': 6.483}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 1.033, 'comp1.Lmp1.ElecUse': 3.826, 'CO2': 0.355}, 'total': 5.214}, 'gains': {'objects': {'product': 11.658}, 'total': 11.658}, 'balance': -0.039}
    # (ImplicitFunc pid=2978607) comp1.Plant.headFW 248.05810179226543
    # (ImplicitFunc pid=2978607) comp1.Plant.fractionGroundCover 0.9980187094349836
    # (ImplicitFunc pid=2978607) comp1.Plant.shootDryMatterContent 0.04869410898168396
    # (ImplicitFunc pid=2978607) comp1.Plant.qualityLoss 5.0
    'G19': {
        "duration": 40,
        "heatingTemp_night": tune.quniform(lower=7, upper=13, q=1),
        "heatingTemp_day": tune.quniform(lower=18, upper=25, q=1),
        "CO2_pureCap": 300,
        "CO2_setpoint_night": tune.qrandint(lower=0, upper=50, q=5),
        "CO2_setpoint_day": tune.qrandint(lower=800, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=50, upper=200, q=5),
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=50, upper=200, q=5),
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        "plantDensity": "1 80; 11 60; 19 41; 23 29; 27 20; 34 15"
    },
    'G20': {
        "duration": 40,
        "heatingTemp_night": tune.quniform(lower=7, upper=13, q=1),
        "heatingTemp_day": tune.quniform(lower=18, upper=25, q=1),
        "CO2_pureCap": 300,
        "CO2_setpoint_night": tune.qrandint(lower=0, upper=50, q=5),
        "CO2_setpoint_day": tune.qrandint(lower=800, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=50, upper=200, q=5),
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=50, upper=200, q=5),
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        "plantDensity": "1 70; 10 43; 16 28; 25 22; 29 19; 36 16; 40 15"
    },
    #     Netprofit=-0.698, Config={'duration': 40, 'heatingTemp_night': 8.0, 'heatingTemp_day': 19.0, 'CO2_pureCap': 300, 'CO2_setpoint_night': 25, 'CO2_setpoint_day': 1080, 'CO2_setpoint_lamp': 0, 'light_intensity': 135, 'light_maxIglob': 130, 'scr1_ToutMax': 5, 'vent_startWnd': 55, 'plantDensity': '1 70; 10 43; 16 28; 25 22; 29 19; 36 16; 40 15'}
    # (ImplicitFunc pid=3122995) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.11, 'AverageHeadm2': 27.5, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 1.188}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.863, 'comp1.Lmp1.Costs': 0.582, 'comp1.Scr1.Costs': 0.137, 'comp1.ConCO2.Costs': 0.493, 'spacingSystem': 0.986, 'plants': 3.296}, 'total': 7.357}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.97, 'comp1.Lmp1.ElecUse': 3.978, 'CO2': 0.409}, 'total': 5.357}, 'gains': {'objects': {'product': 12.016}, 'total': 12.016}, 'balance': -0.698}
    # (ImplicitFunc pid=3122995) comp1.Plant.headFW 246.1552060531553
    # (ImplicitFunc pid=3122995) comp1.Plant.fractionGroundCover 0.9588672618730745
    # (ImplicitFunc pid=3122995) comp1.Plant.shootDryMatterContent 0.047915370670628836
    # (ImplicitFunc pid=3122995) comp1.Plant.qualityLoss 9.5

    # standard
    # 'H0': {'duration': 34,
    #     'heatingTemp_night': 12.0,
    #     'heatingTemp_day': 21.0,
    #     'CO2_pureCap': 109,
    #     'CO2_setpoint_night': 50,
    #     'CO2_setpoint_day': 830,
    #     'CO2_setpoint_lamp': 0,
    #     'light_intensity': 165,
    #     'light_maxIglob': 100,
    #     'scr1_ToutMax': 5,
    #     'vent_startWnd': 55,
    #     'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'
    # },

    # Netprofit=0.959, Config={'duration': 34, 'heatingTemp_night': 12.0, 'heatingTemp_day': 21.0, 'CO2_pureCap': 109, 'CO2_setpoint_night': 50, 'CO2_setpoint_day': 830, 'CO2_setpoint_lamp': 0, 'light_intensity': 165, 'light_maxIglob': 100, 'scr1_ToutMax': 5, 'vent_startWnd': 55, 'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'}
    # (ImplicitFunc pid=3081311) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.093, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.0}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.584, 'comp1.Lmp1.Costs': 0.604, 'comp1.Scr1.Costs': 0.116, 'comp1.ConCO2.Costs': 0.152, 'spacingSystem': 0.559, 'plants': 2.951}, 'total': 5.966}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.982, 'comp1.Lmp1.ElecUse': 4.017, 'CO2': 0.239}, 'total': 5.238}, 'gains': {'objects': {'product': 12.163}, 'total': 12.163}, 'balance': 0.959}
    # (ImplicitFunc pid=3081311) comp1.Plant.headFW 248.93372660781472
    # (ImplicitFunc pid=3081311) comp1.Plant.fractionGroundCover 0.9739553433783031
    # (ImplicitFunc pid=3081311) comp1.Plant.shootDryMatterContent 0.048658339399427646
    # (ImplicitFunc pid=3081311) comp1.Plant.qualityLoss 0.0

    ###############PASUSE##################
    'H1': {
        "duration": 34,
        "heatingTemp_night": tune.quniform(lower=7, upper=13, q=1),
        "heatingTemp_day": tune.quniform(lower=18, upper=25, q=1),
        "CO2_pureCap": tune.qrandint(lower=50, upper=250, q=10),
        "CO2_setpoint_night": tune.qrandint(lower=0, upper=100, q=5),
        "CO2_setpoint_day": tune.qrandint(lower=800, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=50, upper=200, q=5),
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=50, upper=200, q=5),
        "scr1_ToutMax": 5,
        "vent_startWnd": 55,
        "plantDensity": "1 70; 8 37; 13 28; 19 20; 27 15"
    },
    'H1RUNNER': {  # 1.891 pause
        'duration': 34,
        'heatingTemp_night': 10.0,
        'heatingTemp_day': 21.0,
        'CO2_pureCap': 102.0,
        'CO2_setpoint_night': 60,
        'CO2_setpoint_day': 980,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 130,
        'light_maxIglob': 185,
        'scr1_ToutMax': 5,
        'vent_startWnd': 55,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'
    },
    'H1BEST': {  # Netprofit=1.919
        'duration': 34,
        'heatingTemp_night': 8.0,
        'heatingTemp_day': 21.0,
        'CO2_pureCap': 130,
        'CO2_setpoint_night': 0,
        'CO2_setpoint_day': 1160,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 155,
        'light_maxIglob': 85,
        'scr1_ToutMax': 5,
        'vent_startWnd': 55,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'
    },
    # Netprofit=1.919, Config={'duration': 34, 'heatingTemp_night': 8.0, 'heatingTemp_day': 21.0, 'CO2_pureCap': 130, 'CO2_setpoint_night': 0, 'CO2_setpoint_day': 1160, 'CO2_setpoint_lamp': 0, 'light_intensity': 155, 'light_maxIglob': 85, 'scr1_ToutMax': 5, 'vent_startWnd': 55, 'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'}
    # (ImplicitFunc pid=3278375) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.093, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.0}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.584, 'comp1.Lmp1.Costs': 0.568, 'comp1.Scr1.Costs': 0.116, 'comp1.ConCO2.Costs': 0.182, 'spacingSystem': 0.559, 'plants': 2.951}, 'total': 5.959}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.784, 'comp1.Lmp1.ElecUse': 3.185, 'CO2': 0.31}, 'total': 4.279}, 'gains': {'objects': {'product': 12.156}, 'total': 12.156}, 'balance': 1.919}
    # (ImplicitFunc pid=3278375) comp1.Plant.headFW 248.87739624988046
    # (ImplicitFunc pid=3278375) comp1.Plant.fractionGroundCover 1.018163418483413
    # (ImplicitFunc pid=3278375) comp1.Plant.shootDryMatterContent 0.048399759159257406
    # (ImplicitFunc pid=3278375) comp1.Plant.qualityLoss 2.5
    'H2': {
        "duration": 34,
        "heatingTemp_night": tune.quniform(lower=7, upper=13, q=1),
        "heatingTemp_day": tune.quniform(lower=18, upper=25, q=1),
        "CO2_pureCap": tune.qrandint(lower=50, upper=250, q=10),
        "CO2_setpoint_night": tune.qrandint(lower=0, upper=100, q=5),
        "CO2_setpoint_day": tune.qrandint(lower=800, upper=1200, q=10),
        "CO2_setpoint_lamp": 0,
        "light_intensity": tune.qrandint(lower=50, upper=200, q=5),
        # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
        # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
        "light_maxIglob": tune.qrandint(lower=50, upper=200, q=5),
        "scr1_ToutMax": tune.qrandint(lower=-10, upper=5, q=0.5),
        "vent_startWnd": 55,
        "plantDensity": "1 70; 8 37; 13 28; 19 20; 27 15"
    },
    'H3': {
        'duration': 34,
        'heatingTemp_night': 8.0,
        'heatingTemp_day': 21.0,
        'CO2_pureCap': 130,
        'CO2_setpoint_night': 0,
        'CO2_setpoint_day': 1160,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 155,
        'light_maxIglob': 85,
        "scr1_ToutMax": tune.qrandint(lower=-10, upper=5, q=1),
        "vent_startWnd": 55,
        "plantDensity": "1 70; 8 37; 13 28; 19 20; 27 15"
    },
    # Netprofit=1.971, Config={'duration': 34, 'heatingTemp_night': 8.0, 'heatingTemp_day': 21.0, 'CO2_pureCap': 130, 'CO2_setpoint_night': 0, 'CO2_setpoint_day': 1160, 'CO2_setpoint_lamp': 0, 'light_intensity': 155, 'light_maxIglob': 85, 'scr1_ToutMax': 1, 'vent_startWnd': 55, 'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'}
    # (ImplicitFunc pid=3331877) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.093, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.0}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.584, 'comp1.Lmp1.Costs': 0.568, 'comp1.Scr1.Costs': 0.116, 'comp1.ConCO2.Costs': 0.182, 'spacingSystem': 0.559, 'plants': 2.951}, 'total': 5.959}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.798, 'comp1.Lmp1.ElecUse': 3.185, 'CO2': 0.309}, 'total': 4.293}, 'gains': {'objects': {'product': 12.222}, 'total': 12.222}, 'balance': 1.971}
    # (ImplicitFunc pid=3331877) comp1.Plant.headFW 249.41451169572187
    # (ImplicitFunc pid=3331877) comp1.Plant.fractionGroundCover 1.0188346464578004
    # (ImplicitFunc pid=3331877) comp1.Plant.shootDryMatterContent 0.0484380991450634
    # (ImplicitFunc pid=3331877) comp1.Plant.qualityLoss 2.5
    'H3BEST': {
        'duration': 34,
        'heatingTemp_night': 8.0,
        'heatingTemp_day': 21.0,
        'CO2_pureCap': 130,
        'CO2_setpoint_night': 0,
        'CO2_setpoint_day': 1160,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 155,
        'light_maxIglob': 85,
        "scr1_ToutMax": 1,
        "vent_startWnd": 55,
        "plantDensity": "1 70; 8 37; 13 28; 19 20; 27 15"
    },
    'H4': {
        'duration': 34,
        'heatingTemp_night': 8.0,
        'heatingTemp_day': 21.0,
        'CO2_pureCap': 130,
        'CO2_setpoint_night': 0,
        'CO2_setpoint_day': 1160,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 155,
        'light_maxIglob': 85,
        'scr1_material': 'scr_Transparent.par',
        # best 17 Netprofit=2.015
        "scr1_ToutMax": tune.qrandint(lower=0, upper=20, q=1),
        "vent_startWnd": 55,
        "plantDensity": "1 70; 8 37; 13 28; 19 20; 27 15"
    },
    # 'scr_Transparent.par'
    # "comp1.screens.scr1.@lightPollutionPrevention", False
    # Netprofit=1.484, Config={'duration': 34, 'heatingTemp_night': 8.0, 'heatingTemp_day': 21.0, 'CO2_pureCap': 130, 'CO2_setpoint_night': 0, 'CO2_setpoint_day': 1160, 'CO2_setpoint_lamp': 0, 'light_intensity': 155, 'light_maxIglob': 85, 'scr1_ToutMax': 14, 'vent_startWnd': 55, 'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'}
    # (ImplicitFunc pid=3373977) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.093, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.0}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.584, 'comp1.Lmp1.Costs': 0.568, 'comp1.Scr1.Costs': 0.116, 'comp1.ConCO2.Costs': 0.182, 'spacingSystem': 0.559, 'plants': 2.951}, 'total': 5.959}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.797, 'comp1.Lmp1.ElecUse': 3.185, 'CO2': 0.301}, 'total': 4.283}, 'gains': {'objects': {'product': 11.726}, 'total': 11.726}, 'balance': 1.484}
    # (ImplicitFunc pid=3373977) comp1.Plant.headFW 245.38082663298212
    # (ImplicitFunc pid=3373977) comp1.Plant.fractionGroundCover 1.0138434844613735
    # (ImplicitFunc pid=3373977) comp1.Plant.shootDryMatterContent 0.04845707523189252
    # (ImplicitFunc pid=3373977) comp1.Plant.qualityLoss 2.5

    # 'scr_Transparent.par'
    # "comp1.screens.scr1.@lightPollutionPrevention", True
    # Netprofit=2.015, Config={'duration': 34, 'heatingTemp_night': 8.0, 'heatingTemp_day': 21.0, 'CO2_pureCap': 130, 'CO2_setpoint_night': 0, 'CO2_setpoint_day': 1160, 'CO2_setpoint_lamp': 0, 'light_intensity': 155, 'light_maxIglob': 85, 'scr1_ToutMax': 17, 'vent_startWnd': 55, 'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'}
    # (ImplicitFunc pid=3380370) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.093, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.0}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.584, 'comp1.Lmp1.Costs': 0.568, 'comp1.Scr1.Costs': 0.116, 'comp1.ConCO2.Costs': 0.182, 'spacingSystem': 0.559, 'plants': 2.951}, 'total': 5.959}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.756, 'comp1.Lmp1.ElecUse': 3.185, 'CO2': 0.308}, 'total': 4.249}, 'gains': {'objects': {'product': 12.223}, 'total': 12.223}, 'balance': 2.015}
    # (ImplicitFunc pid=3380370) comp1.Plant.headFW 249.4201337383139
    # (ImplicitFunc pid=3380370) comp1.Plant.fractionGroundCover 1.0192997410190168
    # (ImplicitFunc pid=3380370) comp1.Plant.shootDryMatterContent 0.048315402792017965
    # (ImplicitFunc pid=3380370) comp1.Plant.qualityLoss 2.5
    'H4BEST': {
        'duration': 34,
        'heatingTemp_night': 8.0,
        'heatingTemp_day': 21.0,
        'CO2_pureCap': 130,
        'CO2_setpoint_night': 0,
        'CO2_setpoint_day': 1160,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 155,
        'light_maxIglob': 85,
        'scr1_material': 'scr_Transparent.par',
        "scr1_ToutMax": 17,
        "vent_startWnd": 55,
        "plantDensity": "1 70; 8 37; 13 28; 19 20; 27 15"
    },
    'H5': {
        'duration': 34,
        'heatingTemp_night': 8.0,
        'heatingTemp_day': 21.0,
        'CO2_pureCap': 130,
        'CO2_setpoint_night': 0,
        'CO2_setpoint_day': 1160,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 155,
        'light_maxIglob': 85,
        'scr1_material': 'scr_Transparent.par',
        "scr1_ToutMax": 17,
        "vent_startWnd": tune.qrandint(lower=0, upper=100, q=5),
        "plantDensity": "1 70; 8 37; 13 28; 19 20; 27 15"
    },
    # Netprofit=2.035, Config={'duration': 34, 'heatingTemp_night': 8.0, 'heatingTemp_day': 21.0, 'CO2_pureCap': 130, 'CO2_setpoint_night': 0, 'CO2_setpoint_day': 1160, 'CO2_setpoint_lamp': 0, 'light_intensity': 155, 'light_maxIglob': 85, 'scr1_material': 'scr_Transparent.par', 'scr1_ToutMax': 17, 'vent_startWnd': 85, 'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'}
    # (ImplicitFunc pid=3393595) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.093, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.0}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.584, 'comp1.Lmp1.Costs': 0.568, 'comp1.Scr1.Costs': 0.116, 'comp1.ConCO2.Costs': 0.182, 'spacingSystem': 0.559, 'plants': 2.951}, 'total': 5.959}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.754, 'comp1.Lmp1.ElecUse': 3.185, 'CO2': 0.307}, 'total': 4.246}, 'gains': {'objects': {'product': 12.24}, 'total': 12.24}, 'balance': 2.035}
    # (ImplicitFunc pid=3393595) comp1.Plant.headFW 249.56026851262553
    # (ImplicitFunc pid=3393595) comp1.Plant.fractionGroundCover 1.0194942237721423
    # (ImplicitFunc pid=3393595) comp1.Plant.shootDryMatterContent 0.048311382588400655
    # (ImplicitFunc pid=3393595) comp1.Plant.qualityLoss 2.5
    'H5BEST': {
        'duration': 34,
        'heatingTemp_night': 8.0,
        'heatingTemp_day': 21.0,
        'CO2_pureCap': 130,
        'CO2_setpoint_night': 0,
        'CO2_setpoint_day': 1160,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 155,
        'light_maxIglob': 85,
        'scr1_material': 'scr_Transparent.par',
        "scr1_ToutMax": 17,
        "vent_startWnd": 85,
        "plantDensity": "1 70; 8 37; 13 28; 19 20; 27 15"
    },
    'H6': {
        'duration': 34,
        'heatingTemp_night': tune.quniform(lower=6, upper=10, q=0.5),
        'heatingTemp_day': tune.quniform(lower=20, upper=23, q=0.5),
        'CO2_pureCap': tune.qrandint(lower=100, upper=150, q=5),
        'CO2_setpoint_night': tune.qrandint(lower=0, upper=30, q=2),
        'CO2_setpoint_day': tune.qrandint(lower=1000, upper=1200, q=10),
        'CO2_setpoint_lamp': 0,
        'light_intensity': tune.qrandint(lower=140, upper=160, q=2),
        'light_maxIglob': tune.qrandint(lower=70, upper=100, q=2),
        'scr1_material': 'scr_Transparent.par',
        "scr1_ToutMax": tune.qrandint(lower=15, upper=20, q=0.5),
        "vent_startWnd": tune.qrandint(lower=70, upper=100, q=2),
        "plantDensity": "1 70; 8 37; 13 28; 19 20; 27 15"
    },
    # Netprofit=2.22, Config={'duration': 34, 'heatingTemp_night': 7.5, 'heatingTemp_day': 22.5, 'CO2_pureCap': 100, 'CO2_setpoint_night': 6, 'CO2_setpoint_day': 1050, 'CO2_setpoint_lamp': 0, 'light_intensity': 148, 'light_maxIglob': 88, 'scr1_material': 'scr_Transparent.par', 'scr1_ToutMax': 18, 'vent_startWnd': 82, 'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'}
    # (ImplicitFunc pid=3405825) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.093, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.0}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.584, 'comp1.Lmp1.Costs': 0.542, 'comp1.Scr1.Costs': 0.116, 'comp1.ConCO2.Costs': 0.14, 'spacingSystem': 0.559, 'plants': 2.951}, 'total': 5.891}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.834, 'comp1.Lmp1.ElecUse': 3.077, 'CO2': 0.272}, 'total': 4.183}, 'gains': {'objects': {'product': 12.294}, 'total': 12.294}, 'balance': 2.22}
    # (ImplicitFunc pid=3405825) comp1.Plant.headFW 249.99996854612343
    # (ImplicitFunc pid=3405825) comp1.Plant.fractionGroundCover 1.0263335245860434
    # (ImplicitFunc pid=3405825) comp1.Plant.shootDryMatterContent 0.04794429986840564
    # (ImplicitFunc pid=3405825) comp1.Plant.qualityLoss 2.5
    'H6BEST': {
        'duration': 34,
        'heatingTemp_night': 7.5,
        'heatingTemp_day': 22.5,
        'CO2_pureCap': 100,
        'CO2_setpoint_night': 6,
        'CO2_setpoint_day': 1050,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 148,
        'light_maxIglob': 88,
        'scr1_material': 'scr_Transparent.par',
        'scr1_ToutMax': 18,
        'vent_startWnd': 82,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'
    },
    'H7': {
        'duration': 34,
        'heatingTemp_night': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'CO2_pureCap': tune.qrandint(lower=50, upper=110, q=5),
        'CO2_setpoint_night': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_lamp': 0,
        'light_intensity': tune.qrandint(lower=145, upper=155, q=1),
        'light_maxIglob': tune.qrandint(lower=80, upper=95, q=1),
        'scr1_material': 'scr_Transparent.par',
        "scr1_ToutMax": tune.qrandint(lower=17, upper=19, q=0.5),
        "vent_startWnd": tune.qrandint(lower=72, upper=92, q=1),
        "plantDensity": "1 70; 8 37; 13 28; 19 20; 27 15"
    },
    'H8': {
        'duration': 34,
        'heatingTemp_night': 7.5,
        'heatingTemp_day': 22.5,
        'CO2_pureCap': 100,
        'CO2_setpoint_night': 6,
        'CO2_setpoint_day': 1050,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 148,
        'light_maxIglob': 88,
        'light_intensity': tune.qrandint(lower=145, upper=155, q=1),
        "light_hours": tune.qrandint(lower=0, upper=18, q=1),  # 8
        "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),  # 19.5
        'light_maxIglob': tune.qrandint(lower=80, upper=95, q=1),
        'scr1_material': 'scr_Transparent.par',
        'scr1_ToutMax': 18,
        'vent_startWnd': 82,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'
    },
    'H9': {
        'duration': 34,
        'heatingTemp_night': 7.5,
        'heatingTemp_day': 22.5,
        'CO2_pureCap': 100,
        'CO2_setpoint_night': 6,
        'CO2_setpoint_day': 1050,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 148,
        'light_maxIglob': 88,
        'scr1_material': 'scr_Transparent.par',
        'scr1_ToutMax': 18,
        'vent_startWnd': 82,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15',
        'ventOffset': tune.quniform(lower=0, upper=5, q=0.5)
    },
    'H10': {
        'duration': 34,
        'heatingTemp_night': 7.5,
        'heatingTemp_day': 22.5,
        'CO2_pureCap': 100,
        'CO2_setpoint_night': 6,
        'CO2_setpoint_day': 1050,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 148,
        'light_maxIglob': 88,
        'scr1_material': 'scr_Transparent.par',
        'scr1_ToutMax': 18,
        'vent_startWnd': 82,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15',
        'scr1.@enabled': tune.choice([True, False]),
        'scr1.@material': tune.choice(['scr_Transparent.par', 'scr_Shade.par', 'scr_Blackout.par']),
        'scr1.@lightPollutionPrevention': tune.choice([True, False]),
        'scr2.@enabled': tune.choice([True, False]),
        'scr2.@material': tune.choice(['scr_Transparent.par', 'scr_Shade.par', 'scr_Blackout.par']),
        'scr2.@lightPollutionPrevention': tune.choice([True, False]),
    },
    'H11': {
        'duration': 34,
        'heatingTemp_night': 7.5,
        'heatingTemp_day': 22.5,
        'CO2_pureCap': 100,
        'CO2_setpoint_night': 6,
        'CO2_setpoint_day': 1050,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 148,
        'light_maxIglob': 88,
        'scr1_material': 'scr_Transparent.par',
        'scr1_ToutMax': 18,
        'vent_startWnd': 82,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15',
        "closeBelow": tune.qrandint(lower=0, upper=200, q=10),
        "closeAbove": tune.qrandint(lower=500, upper=1500, q=120),
    },
    'H12': {
        'duration': 34,
        'heatingTemp_night_0': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_0': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_1': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_1': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_2': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_2': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_3': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_3': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_4': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_4': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_5': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_5': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_6': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_6': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_7': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_7': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_8': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_8': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_9': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_9': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_10': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_10': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_11': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_11': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_12': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_12': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_13': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_13': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_14': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_14': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_15': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_15': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_16': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_16': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_17': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_17': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_18': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_18': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_19': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_19': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_20': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_20': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_21': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_21': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_22': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_22': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_23': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_23': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_24': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_24': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_25': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_25': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_26': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_26': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_27': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_27': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_28': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_28': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_29': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_29': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_30': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_30': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_31': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_31': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_32': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_32': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'heatingTemp_night_33': tune.quniform(lower=6.5, upper=8.5, q=0.5),
        'heatingTemp_day_33': tune.quniform(lower=21.5, upper=23.5, q=0.5),
        'CO2_pureCap': 100,
        'CO2_setpoint_night': 6,
        'CO2_setpoint_day': 1050,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 148,
        'light_maxIglob': 88,
        'scr1_material': 'scr_Transparent.par',
        'scr1_ToutMax': 18,
        'vent_startWnd': 82,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15',
    },
    # (ImplicitFunc pid=3584590) Netprofit=2.222, Config={'duration': 34, 'heatingTemp_night_0': 8.5, 'heatingTemp_day_0': 21.5, 'heatingTemp_night_1': 8.0, 'heatingTemp_day_1': 21.5, 'heatingTemp_night_2': 7.0, 'heatingTemp_day_2': 23.5, 'heatingTemp_night_3': 8.0, 'heatingTemp_day_3': 22.0, 'heatingTemp_night_4': 7.5, 'heatingTemp_day_4': 21.5, 'heatingTemp_night_5': 7.5, 'heatingTemp_day_5': 21.5, 'heatingTemp_night_6': 7.5, 'heatingTemp_day_6': 22.5, 'heatingTemp_night_7': 8.0, 'heatingTemp_day_7': 21.5, 'heatingTemp_night_8': 8.0, 'heatingTemp_day_8': 22.5, 'heatingTemp_night_9': 6.5, 'heatingTemp_day_9': 23.5, 'heatingTemp_night_10': 7.0, 'heatingTemp_day_10': 23.5, 'heatingTemp_night_11': 7.0, 'heatingTemp_day_11': 22.5, 'heatingTemp_night_12': 6.5, 'heatingTemp_day_12': 22.0, 'heatingTemp_night_13': 8.0, 'heatingTemp_day_13': 22.0, 'heatingTemp_night_14': 8.0, 'heatingTemp_day_14': 22.0, 'heatingTemp_night_15': 7.5, 'heatingTemp_day_15': 22.5, 'heatingTemp_night_16': 6.5, 'heatingTemp_day_16': 22.5, 'heatingTemp_night_17': 7.0, 'heatingTemp_day_17': 23.0, 'heatingTemp_night_18': 8.5, 'heatingTemp_day_18': 23.5, 'heatingTemp_night_19': 6.5, 'heatingTemp_day_19': 21.5, 'heatingTemp_night_20': 7.5, 'heatingTemp_day_20': 23.5, 'heatingTemp_night_21': 6.5, 'heatingTemp_day_21': 22.5, 'heatingTemp_night_22': 8.5, 'heatingTemp_day_22': 22.0, 'heatingTemp_night_23': 8.0, 'heatingTemp_day_23': 22.5, 'heatingTemp_night_24': 7.5, 'heatingTemp_day_24': 22.0, 'heatingTemp_night_25': 7.5, 'heatingTemp_day_25': 23.0, 'heatingTemp_night_26': 7.0, 'heatingTemp_day_26': 21.5, 'heatingTemp_night_27': 6.5, 'heatingTemp_day_27': 23.5, 'heatingTemp_night_28': 7.0, 'heatingTemp_day_28': 22.5, 'heatingTemp_night_29': 6.5, 'heatingTemp_day_29': 22.5, 'heatingTemp_night_30': 7.5, 'heatingTemp_day_30': 22.0, 'heatingTemp_night_31': 8.0, 'heatingTemp_day_31': 22.5, 'heatingTemp_night_32': 8.5, 'heatingTemp_day_32': 22.5, 'heatingTemp_night_33': 7.5, 'heatingTemp_day_33': 21.5, 'CO2_pureCap': 100, 'CO2_setpoint_night': 6, 'CO2_setpoint_day': 1050, 'CO2_setpoint_lamp': 0, 'light_intensity': 148, 'light_maxIglob': 88, 'scr1_material': 'scr_Transparent.par', 'scr1_ToutMax': 18, 'vent_startWnd': 82, 'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'}
    # (ImplicitFunc pid=3584590) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.093, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.0}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.584, 'comp1.Lmp1.Costs': 0.542, 'comp1.Scr1.Costs': 0.116, 'comp1.ConCO2.Costs': 0.14, 'spacingSystem': 0.559, 'plants': 2.951}, 'total': 5.891}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.83, 'comp1.Lmp1.ElecUse': 3.077, 'CO2': 0.272}, 'total': 4.179}, 'gains': {'objects': {'product': 12.293}, 'total': 12.293}, 'balance': 2.222}
    # (ImplicitFunc pid=3584590) comp1.Plant.headFW 249.98586863016394
    # (ImplicitFunc pid=3584590) comp1.Plant.fractionGroundCover 1.0263141096954633
    # (ImplicitFunc pid=3584590) comp1.Plant.shootDryMatterContent 0.047964345593168735
    # (ImplicitFunc pid=3584590) comp1.Plant.qualityLoss 2.5
    'H12BEST': {
        'duration': 34,
        'heatingTemp_night_0': 8.5,
        'heatingTemp_day_0': 21.5,
        'heatingTemp_night_1': 8.0,
        'heatingTemp_day_1': 21.5,
        'heatingTemp_night_2': 7.0,
        'heatingTemp_day_2': 23.5,
        'heatingTemp_night_3': 8.0,
        'heatingTemp_day_3': 22.0,
        'heatingTemp_night_4': 7.5,
        'heatingTemp_day_4': 21.5,
        'heatingTemp_night_5': 7.5,
        'heatingTemp_day_5': 21.5,
        'heatingTemp_night_6': 7.5,
        'heatingTemp_day_6': 22.5,
        'heatingTemp_night_7': 8.0,
        'heatingTemp_day_7': 21.5,
        'heatingTemp_night_8': 8.0,
        'heatingTemp_day_8': 22.5,
        'heatingTemp_night_9': 6.5,
        'heatingTemp_day_9': 23.5,
        'heatingTemp_night_10': 7.0,
        'heatingTemp_day_10': 23.5,
        'heatingTemp_night_11': 7.0,
        'heatingTemp_day_11': 22.5,
        'heatingTemp_night_12': 6.5,
        'heatingTemp_day_12': 22.0,
        'heatingTemp_night_13': 8.0,
        'heatingTemp_day_13': 22.0,
        'heatingTemp_night_14': 8.0,
        'heatingTemp_day_14': 22.0,
        'heatingTemp_night_15': 7.5,
        'heatingTemp_day_15': 22.5,
        'heatingTemp_night_16': 6.5,
        'heatingTemp_day_16': 22.5,
        'heatingTemp_night_17': 7.0,
        'heatingTemp_day_17': 23.0,
        'heatingTemp_night_18': 8.5,
        'heatingTemp_day_18': 23.5,
        'heatingTemp_night_19': 6.5,
        'heatingTemp_day_19': 21.5,
        'heatingTemp_night_20': 7.5,
        'heatingTemp_day_20': 23.5,
        'heatingTemp_night_21': 6.5,
        'heatingTemp_day_21': 22.5,
        'heatingTemp_night_22': 8.5,
        'heatingTemp_day_22': 22.0,
        'heatingTemp_night_23': 8.0,
        'heatingTemp_day_23': 22.5,
        'heatingTemp_night_24': 7.5,
        'heatingTemp_day_24': 22.0,
        'heatingTemp_night_25': 7.5,
        'heatingTemp_day_25': 23.0,
        'heatingTemp_night_26': 7.0,
        'heatingTemp_day_26': 21.5,
        'heatingTemp_night_27': 6.5,
        'heatingTemp_day_27': 23.5,
        'heatingTemp_night_28': 7.0,
        'heatingTemp_day_28': 22.5,
        'heatingTemp_night_29': 6.5,
        'heatingTemp_day_29': 22.5,
        'heatingTemp_night_30': 7.5,
        'heatingTemp_day_30': 22.0,
        'heatingTemp_night_31': 8.0,
        'heatingTemp_day_31': 22.5,
        'heatingTemp_night_32': 8.5,
        'heatingTemp_day_32': 22.5,
        'heatingTemp_night_33': 7.5,
        'heatingTemp_day_33': 21.5,
        'CO2_pureCap': 100,
        'CO2_setpoint_night': 6,
        'CO2_setpoint_day': 1050,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 148,
        'light_maxIglob': 88,
        'scr1_material': 'scr_Transparent.par',
        'scr1_ToutMax': 18,
        'vent_startWnd': 82,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'
    },
    'H13': {
        'duration': 34,
        'heatingTemp_night_0': tune.quniform(lower=7.5, upper=9.5, q=0.5),
        'heatingTemp_day_0': tune.quniform(lower=20.5, upper=22.5, q=0.5),
        'heatingTemp_night_1': 8.0,
        'heatingTemp_day_1': tune.quniform(lower=20.5, upper=22.5, q=0.5),
        'heatingTemp_night_2': 7.0,
        'heatingTemp_day_2': tune.quniform(lower=22.5, upper=24.5, q=0.5),
        'heatingTemp_night_3': 8.0,
        'heatingTemp_day_3': 22.0,
        'heatingTemp_night_4': 7.5,
        'heatingTemp_day_4': tune.quniform(lower=20.5, upper=22.5, q=0.5),
        'heatingTemp_night_5': 7.5,
        'heatingTemp_day_5': tune.quniform(lower=20.5, upper=22.5, q=0.5),
        'heatingTemp_night_6': 7.5,
        'heatingTemp_day_6': 22.5,
        'heatingTemp_night_7': 8.0,
        'heatingTemp_day_7': tune.quniform(lower=20.5, upper=22.5, q=0.5),
        'heatingTemp_night_8': 8.0,
        'heatingTemp_day_8': 22.5,
        'heatingTemp_night_9': tune.quniform(lower=5.5, upper=7.5, q=0.5),
        'heatingTemp_day_9': tune.quniform(lower=22.5, upper=24.5, q=0.5),
        'heatingTemp_night_10': 7.0,
        'heatingTemp_day_10': tune.quniform(lower=22.5, upper=24.5, q=0.5),
        'heatingTemp_night_11': 7.0,
        'heatingTemp_day_11': 22.5,
        'heatingTemp_night_12': tune.quniform(lower=5.5, upper=7.5, q=0.5),
        'heatingTemp_day_12': 22.0,
        'heatingTemp_night_13': 8.0,
        'heatingTemp_day_13': 22.0,
        'heatingTemp_night_14': 8.0,
        'heatingTemp_day_14': 22.0,
        'heatingTemp_night_15': 7.5,
        'heatingTemp_day_15': 22.5,
        'heatingTemp_night_16': tune.quniform(lower=5.5, upper=7.5, q=0.5),
        'heatingTemp_day_16': 22.5,
        'heatingTemp_night_17': 7.0,
        'heatingTemp_day_17': 23.0,
        'heatingTemp_night_18': tune.quniform(lower=7.5, upper=9.5, q=0.5),
        'heatingTemp_day_18': tune.quniform(lower=22.5, upper=24.5, q=0.5),
        'heatingTemp_night_19': tune.quniform(lower=5.5, upper=7.5, q=0.5),
        'heatingTemp_day_19': tune.quniform(lower=20.5, upper=22.5, q=0.5),
        'heatingTemp_night_20': 7.5,
        'heatingTemp_day_20': tune.quniform(lower=22.5, upper=24.5, q=0.5),
        'heatingTemp_night_21': tune.quniform(lower=5.5, upper=7.5, q=0.5),
        'heatingTemp_day_21': 22.5,
        'heatingTemp_night_22': tune.quniform(lower=7.5, upper=9.5, q=0.5),
        'heatingTemp_day_22': 22.0,
        'heatingTemp_night_23': 8.0,
        'heatingTemp_day_23': 22.5,
        'heatingTemp_night_24': 7.5,
        'heatingTemp_day_24': 22.0,
        'heatingTemp_night_25': 7.5,
        'heatingTemp_day_25': 23.0,
        'heatingTemp_night_26': 7.0,
        'heatingTemp_day_26': tune.quniform(lower=20.5, upper=22.5, q=0.5),
        'heatingTemp_night_27': tune.quniform(lower=5.5, upper=7.5, q=0.5),
        'heatingTemp_day_27': tune.quniform(lower=22.5, upper=24.5, q=0.5),
        'heatingTemp_night_28': 7.0,
        'heatingTemp_day_28': 22.5,
        'heatingTemp_night_29': tune.quniform(lower=5.5, upper=7.5, q=0.5),
        'heatingTemp_day_29': 22.5,
        'heatingTemp_night_30': 7.5,
        'heatingTemp_day_30': 22.0,
        'heatingTemp_night_31': 8.0,
        'heatingTemp_day_31': 22.5,
        'heatingTemp_night_32': tune.quniform(lower=7.5, upper=9.5, q=0.5),
        'heatingTemp_day_32': 22.5,
        'heatingTemp_night_33': 7.5,
        'heatingTemp_day_33': tune.quniform(lower=20.5, upper=22.5, q=0.5),
        'CO2_pureCap': 100,
        'CO2_setpoint_night': 6,
        'CO2_setpoint_day': 1050,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 148,
        'light_maxIglob': 88,
        'scr1_material': 'scr_Transparent.par',
        'scr1_ToutMax': 18,
        'vent_startWnd': 82,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'
    },
    'H13BEST': {  # 2.222
        'duration': 34,
        'heatingTemp_night_0': 8.5,
        'heatingTemp_day_0': 21.5,
        'heatingTemp_night_1': 8.0,
        'heatingTemp_day_1': 21.5,
        'heatingTemp_night_2': 7.0,
        'heatingTemp_day_2': 23.5,
        'heatingTemp_night_3': 8.0,
        'heatingTemp_day_3': 22.0,
        'heatingTemp_night_4': 7.5,
        'heatingTemp_day_4': 21.5,
        'heatingTemp_night_5': 7.5,
        'heatingTemp_day_5': 21.5,
        'heatingTemp_night_6': 7.5,
        'heatingTemp_day_6': 22.5,
        'heatingTemp_night_7': 8.0,
        'heatingTemp_day_7': 21.5,
        'heatingTemp_night_8': 8.0,
        'heatingTemp_day_8': 22.5,
        'heatingTemp_night_9': 6.5,
        'heatingTemp_day_9': 23.5,
        'heatingTemp_night_10': 7.0,
        'heatingTemp_day_10': 23.5,
        'heatingTemp_night_11': 7.0,
        'heatingTemp_day_11': 22.5,
        'heatingTemp_night_12': 6.5,
        'heatingTemp_day_12': 22.0,
        'heatingTemp_night_13': 8.0,
        'heatingTemp_day_13': 22.0,
        'heatingTemp_night_14': 8.0,
        'heatingTemp_day_14': 22.0,
        'heatingTemp_night_15': 7.5,
        'heatingTemp_day_15': 22.5,
        'heatingTemp_night_16': 6.5,
        'heatingTemp_day_16': 22.5,
        'heatingTemp_night_17': 7.0,
        'heatingTemp_day_17': 23.0,
        'heatingTemp_night_18': 8.5,
        'heatingTemp_day_18': 23.5,
        'heatingTemp_night_19': 6.5,
        'heatingTemp_day_19': 21.5,
        'heatingTemp_night_20': 7.5,
        'heatingTemp_day_20': 23.5,
        'heatingTemp_night_21': 6.5,
        'heatingTemp_day_21': 22.5,
        'heatingTemp_night_22': 8.5,
        'heatingTemp_day_22': 22.0,
        'heatingTemp_night_23': 8.0,
        'heatingTemp_day_23': 22.5,
        'heatingTemp_night_24': 7.5,
        'heatingTemp_day_24': 22.0,
        'heatingTemp_night_25': 7.5,
        'heatingTemp_day_25': 23.0,
        'heatingTemp_night_26': 7.0,
        'heatingTemp_day_26': 21.5,
        'heatingTemp_night_27': 6.5,
        'heatingTemp_day_27': 23.5,
        'heatingTemp_night_28': 7.0,
        'heatingTemp_day_28': 22.5,
        'heatingTemp_night_29': 6.5,
        'heatingTemp_day_29': 22.5,
        'heatingTemp_night_30': 7.5,
        'heatingTemp_day_30': 22.0,
        'heatingTemp_night_31': 8.0,
        'heatingTemp_day_31': 22.5,
        'heatingTemp_night_32': 8.5,
        'heatingTemp_day_32': 22.5,
        'heatingTemp_night_33': 7.5,
        'heatingTemp_day_33': 21.5,
        'CO2_pureCap': 100,
        'CO2_setpoint_night': 6,
        'CO2_setpoint_day': 1050,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 148,
        'light_maxIglob': 88,
        'scr1_material': 'scr_Transparent.par',
        'scr1_ToutMax': 18,
        'vent_startWnd': 82,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15'
    },
    'H14': {
        'duration': 34,
        'heatingTemp_night_0': 8.5,
        'heatingTemp_day_0': 21.5,
        'heatingTemp_night_1': 8.0,
        'heatingTemp_day_1': 21.5,
        'heatingTemp_night_2': 7.0,
        'heatingTemp_day_2': 23.5,
        'heatingTemp_night_3': 8.0,
        'heatingTemp_day_3': 22.0,
        'heatingTemp_night_4': 7.5,
        'heatingTemp_day_4': 21.5,
        'heatingTemp_night_5': 7.5,
        'heatingTemp_day_5': 21.5,
        'heatingTemp_night_6': 7.5,
        'heatingTemp_day_6': 22.5,
        'heatingTemp_night_7': 8.0,
        'heatingTemp_day_7': 21.5,
        'heatingTemp_night_8': 8.0,
        'heatingTemp_day_8': 22.5,
        'heatingTemp_night_9': 6.5,
        'heatingTemp_day_9': 23.5,
        'heatingTemp_night_10': 7.0,
        'heatingTemp_day_10': 23.5,
        'heatingTemp_night_11': 7.0,
        'heatingTemp_day_11': 22.5,
        'heatingTemp_night_12': 6.5,
        'heatingTemp_day_12': 22.0,
        'heatingTemp_night_13': 8.0,
        'heatingTemp_day_13': 22.0,
        'heatingTemp_night_14': 8.0,
        'heatingTemp_day_14': 22.0,
        'heatingTemp_night_15': 7.5,
        'heatingTemp_day_15': 22.5,
        'heatingTemp_night_16': 6.5,
        'heatingTemp_day_16': 22.5,
        'heatingTemp_night_17': 7.0,
        'heatingTemp_day_17': 23.0,
        'heatingTemp_night_18': 8.5,
        'heatingTemp_day_18': 23.5,
        'heatingTemp_night_19': 6.5,
        'heatingTemp_day_19': 21.5,
        'heatingTemp_night_20': 7.5,
        'heatingTemp_day_20': 23.5,
        'heatingTemp_night_21': 6.5,
        'heatingTemp_day_21': 22.5,
        'heatingTemp_night_22': 8.5,
        'heatingTemp_day_22': 22.0,
        'heatingTemp_night_23': 8.0,
        'heatingTemp_day_23': 22.5,
        'heatingTemp_night_24': 7.5,
        'heatingTemp_day_24': 22.0,
        'heatingTemp_night_25': 7.5,
        'heatingTemp_day_25': 23.0,
        'heatingTemp_night_26': 7.0,
        'heatingTemp_day_26': 21.5,
        'heatingTemp_night_27': 6.5,
        'heatingTemp_day_27': 23.5,
        'heatingTemp_night_28': 7.0,
        'heatingTemp_day_28': 22.5,
        'heatingTemp_night_29': 6.5,
        'heatingTemp_day_29': 22.5,
        'heatingTemp_night_30': 7.5,
        'heatingTemp_day_30': 22.0,
        'heatingTemp_night_31': 8.0,
        'heatingTemp_day_31': 22.5,
        'heatingTemp_night_32': 8.5,
        'heatingTemp_day_32': 22.5,
        'heatingTemp_night_33': 7.5,
        'heatingTemp_day_33': 21.5,
        'CO2_pureCap': 100,
        # 'CO2_setpoint_night': 6,
        # 'CO2_setpoint_day': 1050,
        'CO2_setpoint_night_0': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_0': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_1': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_1': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_2': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_2': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_3': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_3': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_4': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_4': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_5': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_5': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_6': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_6': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_7': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_7': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_8': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_8': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_9': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_9': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_10': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_10': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_11': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_11': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_12': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_12': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_13': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_13': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_14': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_14': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_15': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_15': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_16': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_16': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_17': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_17': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_18': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_18': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_19': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_19': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_20': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_20': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_21': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_21': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_22': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_22': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_23': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_23': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_24': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_24': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_25': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_25': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_26': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_26': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_27': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_27': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_28': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_28': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_29': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_29': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_30': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_30': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_31': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_31': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_32': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_32': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_night_33': tune.qrandint(lower=3, upper=10, q=1),
        'CO2_setpoint_day_33': tune.qrandint(lower=1000, upper=1100, q=5),
        'CO2_setpoint_lamp': 0,
        'light_intensity': 148,
        'light_maxIglob': 88,
        'scr1_material': 'scr_Transparent.par',
        'scr1_ToutMax': 18,
        'vent_startWnd': 82,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15',
    },
    # {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.093, 'AverageHeadm2': 24.6, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.0}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.584, 'comp1.Lmp1.Costs': 0.542, 'comp1.Scr1.Costs': 0.116, 'comp1.ConCO2.Costs': 0.14, 'spacingSystem': 0.559, 'plants': 2.951}, 'total': 5.891}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.83, 'comp1.Lmp1.ElecUse': 3.077, 'CO2': 0.272}, 'total': 4.179}, 'gains': {'objects': {'product': 12.294}, 'total': 12.294}, 'balance': 2.224}
    # (ImplicitFunc pid=3632018) comp1.Plant.headFW 249.99962156174664
    # (ImplicitFunc pid=3632018) comp1.Plant.fractionGroundCover 1.026332703541575
    # (ImplicitFunc pid=3632018) comp1.Plant.shootDryMatterContent 0.04796470937627494
    # (ImplicitFunc pid=3632018) comp1.Plant.qualityLoss 2.5
    'H14BESTRESUME': {
        'duration': 35,
        'heatingTemp_night_0': 8.5,
        'heatingTemp_day_0': 21.5,
        'heatingTemp_night_1': 8.0,
        'heatingTemp_day_1': 21.5,
        'heatingTemp_night_2': 7.0,
        'heatingTemp_day_2': 23.5,
        'heatingTemp_night_3': 8.0,
        'heatingTemp_day_3': 22.0,
        'heatingTemp_night_4': 7.5,
        'heatingTemp_day_4': 21.5,
        'heatingTemp_night_5': 7.5,
        'heatingTemp_day_5': 21.5,
        'heatingTemp_night_6': 7.5,
        'heatingTemp_day_6': 22.5,
        'heatingTemp_night_7': 8.0,
        'heatingTemp_day_7': 21.5,
        'heatingTemp_night_8': 8.0,
        'heatingTemp_day_8': 22.5,
        'heatingTemp_night_9': 6.5,
        'heatingTemp_day_9': 23.5,
        'heatingTemp_night_10': 7.0,
        'heatingTemp_day_10': 23.5,
        'heatingTemp_night_11': 7.0,
        'heatingTemp_day_11': 22.5,
        'heatingTemp_night_12': 6.5,
        'heatingTemp_day_12': 22.0,
        'heatingTemp_night_13': 8.0,
        'heatingTemp_day_13': 22.0,
        'heatingTemp_night_14': 8.0,
        'heatingTemp_day_14': 22.0,
        'heatingTemp_night_15': 7.5,
        'heatingTemp_day_15': 22.5,
        'heatingTemp_night_16': 6.5,
        'heatingTemp_day_16': 22.5,
        'heatingTemp_night_17': 7.0,
        'heatingTemp_day_17': 23.0,
        'heatingTemp_night_18': 8.5,
        'heatingTemp_day_18': 23.5,
        'heatingTemp_night_19': 6.5,
        'heatingTemp_day_19': 21.5,
        'heatingTemp_night_20': 7.5,
        'heatingTemp_day_20': 23.5,
        'heatingTemp_night_21': 6.5,
        'heatingTemp_day_21': 22.5,
        'heatingTemp_night_22': 8.5,
        'heatingTemp_day_22': 22.0,
        'heatingTemp_night_23': 8.0,
        'heatingTemp_day_23': 22.5,
        'heatingTemp_night_24': 7.5,
        'heatingTemp_day_24': 22.0,
        'heatingTemp_night_25': 7.5,
        'heatingTemp_day_25': 23.0,
        'heatingTemp_night_26': 7.0,
        'heatingTemp_day_26': 21.5,
        'heatingTemp_night_27': 6.5,
        'heatingTemp_day_27': 23.5,
        'heatingTemp_night_28': 7.0,
        'heatingTemp_day_28': 22.5,
        'heatingTemp_night_29': 6.5,
        'heatingTemp_day_29': 22.5,
        'heatingTemp_night_30': 7.5,
        'heatingTemp_day_30': 22.0,
        'heatingTemp_night_31': 8.0,
        'heatingTemp_day_31': 22.5,
        'heatingTemp_night_32': 8.5,
        'heatingTemp_day_32': 22.5,
        'heatingTemp_night_33': 7.5,
        'heatingTemp_day_33': 21.5,
        'heatingTemp_night_34': 7.5,
        'heatingTemp_day_34': 21.5,
        'CO2_pureCap': 100,
        # 'CO2_setpoint_night': 6,
        # 'CO2_setpoint_day': 1050,
        'CO2_setpoint_night_0': 4, 'CO2_setpoint_day_0': 1040, 'CO2_setpoint_night_1': 5, 'CO2_setpoint_day_1': 1020, 'CO2_setpoint_night_2': 6, 'CO2_setpoint_day_2': 1070, 'CO2_setpoint_night_3': 5, 'CO2_setpoint_day_3': 1015, 'CO2_setpoint_night_4': 9, 'CO2_setpoint_day_4': 1000, 'CO2_setpoint_night_5': 8, 'CO2_setpoint_day_5': 1060, 'CO2_setpoint_night_6': 7, 'CO2_setpoint_day_6': 1065, 'CO2_setpoint_night_7': 5, 'CO2_setpoint_day_7': 1065, 'CO2_setpoint_night_8': 5, 'CO2_setpoint_day_8': 1070, 'CO2_setpoint_night_9': 7, 'CO2_setpoint_day_9': 1065, 'CO2_setpoint_night_10': 6, 'CO2_setpoint_day_10': 1080, 'CO2_setpoint_night_11': 4, 'CO2_setpoint_day_11': 1085, 'CO2_setpoint_night_12': 8, 'CO2_setpoint_day_12': 1020, 'CO2_setpoint_night_13': 3, 'CO2_setpoint_day_13': 1075, 'CO2_setpoint_night_14': 4, 'CO2_setpoint_day_14': 1080, 'CO2_setpoint_night_15': 7, 'CO2_setpoint_day_15': 1015, 'CO2_setpoint_night_16': 4, 'CO2_setpoint_day_16': 1045, 'CO2_setpoint_night_17': 7, 'CO2_setpoint_day_17': 1100, 'CO2_setpoint_night_18': 4, 'CO2_setpoint_day_18': 1030, 'CO2_setpoint_night_19': 6, 'CO2_setpoint_day_19': 1100, 'CO2_setpoint_night_20': 3, 'CO2_setpoint_day_20': 1070, 'CO2_setpoint_night_21': 7, 'CO2_setpoint_day_21': 1070, 'CO2_setpoint_night_22': 5, 'CO2_setpoint_day_22': 1065, 'CO2_setpoint_night_23': 6, 'CO2_setpoint_day_23': 1055, 'CO2_setpoint_night_24': 6, 'CO2_setpoint_day_24': 1020, 'CO2_setpoint_night_25': 5, 'CO2_setpoint_day_25': 1035, 'CO2_setpoint_night_26': 4, 'CO2_setpoint_day_26': 1050, 'CO2_setpoint_night_27': 4, 'CO2_setpoint_day_27': 1005, 'CO2_setpoint_night_28': 4, 'CO2_setpoint_day_28': 1035, 'CO2_setpoint_night_29': 5, 'CO2_setpoint_day_29': 1015, 'CO2_setpoint_night_30': 5, 'CO2_setpoint_day_30': 1050, 'CO2_setpoint_night_31': 6, 'CO2_setpoint_day_31': 1055, 'CO2_setpoint_night_32': 7, 'CO2_setpoint_day_32': 1055, 'CO2_setpoint_night_33': 4, 'CO2_setpoint_day_33': 1035, 'CO2_setpoint_night_34': 4, 'CO2_setpoint_day_34': 1035,
        'CO2_setpoint_lamp': 0,
        'light_intensity': 148,
        'light_maxIglob': 88,
        'scr1_material': 'scr_Transparent.par',
        'scr1_ToutMax': 18,
        'vent_startWnd': 82,
        'plantDensity': '1 70; 8 37; 13 28; 19 20; 27 15',
        # "lamp_type": "lmp_LED32.par", #lmp_SON-T1000W.par, lmp_LED29.par, lmp_LED32.par
    },

    # 'H9': {
    #         "duration": 33,
    #         "heatingTemp_night": tune.quniform(lower=7, upper=13, q=1),
    #         "heatingTemp_day": tune.quniform(lower=18, upper=25, q=1),
    #         "CO2_pureCap": tune.qrandint(lower=50, upper=250, q=10),
    #         "CO2_setpoint_night": tune.qrandint(lower=0, upper=100, q=5),
    #         "CO2_setpoint_day": tune.qrandint(lower=800, upper=1200, q=10),
    #         "CO2_setpoint_lamp": 0,
    #         "light_intensity": tune.qrandint(lower=50, upper=200, q=5),
    #         # "light_hours": tune.qrandint(lower=0, upper=18, q=1),
    #         # "light_endTime": tune.quniform(lower=18, upper=20, q=0.5),
    #         "light_maxIglob": tune.qrandint(lower=50, upper=200, q=5),
    #         "scr1_ToutMax": 5,
    #         "vent_startWnd": 55,
    #         "plantDensity": "1 74; 7 36; 15 24; 21 18; 29 16; 32 15"
    #     },
    # #     Netprofit=1.629, Config={'duration': 33, 'heatingTemp_night': 7.0, 'heatingTemp_day': 21.0, 'CO2_pureCap': 100, 'CO2_setpoint_night': 25, 'CO2_setpoint_day': 1160, 'CO2_setpoint_lamp': 0, 'light_intensity': 165, 'light_maxIglob': 145, 'scr1_ToutMax': 5, 'vent_startWnd': 55, 'plantDensity': '1 74; 7 36; 15 24; 21 18; 29 16; 32 15'}
    # # (ImplicitFunc pid=3271745) {'info': {'unit': 'euro/m2', 'fractionOfYear': 0.09, 'AverageHeadm2': 25.0, 'bonusMalusOnDMC': 1.0, 'lostByLowQuality': 0.0}, 'fixedCosts': {'objects': {'comp1.Greenhouse.Costs': 1.537, 'comp1.Lmp1.Costs': 0.587, 'comp1.Scr1.Costs': 0.113, 'comp1.ConCO2.Costs': 0.136, 'spacingSystem': 0.678, 'plants': 3.003}, 'total': 6.054}, 'variableCosts': {'objects': {'comp1.Pipe1.GasUse': 0.685, 'comp1.Lmp1.ElecUse': 3.768, 'CO2': 0.284}, 'total': 4.737}, 'gains': {'objects': {'product': 12.42}, 'total': 12.42}, 'balance': 1.629}
    # # (ImplicitFunc pid=3271745) comp1.Plant.headFW 250.74859696266353
    # # (ImplicitFunc pid=3271745) comp1.Plant.fractionGroundCover 0.9642100646631628
    # # (ImplicitFunc pid=3271745) comp1.Plant.shootDryMatterContent 0.04858645130156503
    # # (ImplicitFunc pid=3271745) comp1.Plant.qualityLoss 0.0
    # 'H10': {
    #         'duration': 34,
    #         'heatingTemp_night': 8.0,
    #         'heatingTemp_day': 21.0,
    #         'CO2_pureCap': 130,
    #         'CO2_setpoint_night': 0,
    #         'CO2_setpoint_day': 1160,
    #         'CO2_setpoint_lamp': 0,
    #         'light_intensity': 155,
    #         'light_maxIglob': 85,
    #         'scr1_material': 'scr_Transparent.par',
    #         "scr1_ToutMax": 17,
    #         "vent_startWnd": 85,
    #         "plantDensity": tune.choice(make_plant_density(34))
    #     },
}
# for i in range(34):
#     SPACES['H12'][f'heatingTemp_night_{i}'] = tune.quniform(lower=6.5, upper=8.5, q=0.5),
#     SPACES['H12'][f'heatingTemp_day_{i}'] = tune.quniform(lower=21.5, upper=23.5, q=0.5),
