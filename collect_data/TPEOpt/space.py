from ray import tune


def make_plant_density():
    # TODO: implement randomly generating plant density strings
    return [
        "1 80; 11 45; 19 25; 27 15",
        "1 90; 7 60; 14 40; 21 30; 28 20; 34 15",
        "1 80; 9 50; 14 25; 20 20; 27 15",
    ]


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
        "plantDensity": tune.choice(make_plant_density()),
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
        "plantDensity": tune.choice(make_plant_density()),
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
}


# best config: {'num_days': 37, 'heatingTemp_night': 10.36368083752856, 'heatingTemp_day': 19.31021249899327, 'CO2_pureCap': 190, 'CO2_setpoint_night': 670, 'CO2_setpoint_day': 1050, 'CO2_setpoint_lamp': 0, 'light_intensity': 50, 'light_hours': 10.0, 'light_endTime': 18.0, 'light_maxIglob': 295, 'plantDensity': '1 80; 11 45; 19 25; 27 15'}
# best netprofit: 6.713

# 2021-07-12 20:44:44,039 INFO tune.py:549 -- Total run time: 2504.97 seconds (2504.25 seconds for the tuning loop).
# best config: {'num_days': 37, 'heatingTemp_night': 10.3, 'heatingTemp_day': 19.400000000000002, 'CO2_pureCap': 200, 'CO2_setpoint_night': 690, 'CO2_setpoint_day': 1110, 'CO2_setpoint_lamp': 0, 'light_intensity': 40, 'light_hours': 7.800000000000001, 'light_endTime': 19.3, 'light_maxIglob': 300, 'plantDensity': '1 80; 11 45; 19 25; 27 15'}
# best netprofit: 6.87