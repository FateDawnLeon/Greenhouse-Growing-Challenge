import warnings
warnings.filterwarnings('ignore')

from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.utils import use_named_args, dump
from skopt.space import Integer, Real, Categorical

from pprint import pprint
from utils import ControlParamSimple
from run_simulation import try_on_simulator
import numpy as np
import random
import copy
import pytz
from datetime import datetime
import sys
import glob
import os


DIMS = {
    'A1': [
        # runtime: method=gbrt_init=1000_ncall=10000_RS=42_SP=A_P=1_sim=A
        # Best Parameters: [37, 29.331173428460527, 19.740135016200067, 105.54579140908399, 735.4649817201819, 800.7794533665503, 1114.7327640161704, 10.831808389882115, 12.963267259805608, 16.377776700064487, 347.19371387181457]
        # Best NetProfit: 3.923
        Integer(name='num_days', low=30, high=50),
        Real(name='heatingTemp_night', low=15, high=30),
        Real(name='heatingTemp_day', low=5, high=20),
        Real(name='CO2_pureCap', low=100, high=200),
        Real(name='CO2_setpoint_night', low=400, high=800),
        Real(name='CO2_setpoint_day', low=800, high=1200),
        Real(name='CO2_setpoint_lamp', low=800, high=1200),
        Real(name='light_intensity', low=0, high=200),
        Real(name='light_hours', low=0, high=20),
        Real(name='light_endTime', low=0, high=24),
        Real(name='light_maxIglob', low=100, high=400),
    ],
    'A2': [
        Integer(name='num_days', low=35, high=45),
        Real(name='heatingTemp_night', low=10, high=30),
        Real(name='heatingTemp_day', low=10, high=20),
        Real(name='CO2_pureCap', low=100, high=200),
        Real(name='CO2_setpoint_night', low=400, high=1200),
        Real(name='CO2_setpoint_day', low=400, high=1200),
        Real(name='CO2_setpoint_lamp', low=400, high=1200),
        Real(name='light_intensity', low=0, high=200),
        Real(name='light_hours', low=4, high=20),
        Real(name='light_endTime', low=0, high=24),
        Real(name='light_maxIglob', low=100, high=200),
    ],
    'A3': [
        Integer(name='num_days', low=30, high=50),
        Integer(name='heatingTemp_night', low=0, high=20),
        Integer(name='heatingTemp_day', low=10, high=30),
        Integer(name='CO2_pureCap', low=100, high=200),
        Integer(name='CO2_setpoint_night', low=0, high=1200),
        Integer(name='CO2_setpoint_day', low=0, high=1200),
        Integer(name='CO2_setpoint_lamp', low=0, high=1200),
        Integer(name='light_intensity', low=0, high=200),
        Integer(name='light_hours', low=0, high=24),
        Integer(name='light_endTime', low=0, high=24),
        Integer(name='light_maxIglob', low=100, high=300),
    ],
    'A4': [
        # Best Parameters: [38, 10, 26, 182, 694, 1200, 912, 0, 12, 24, 300]
        # Best NetProfit: 4.941
        Integer(name='num_days', low=38, high=42),
        Integer(name='heatingTemp_night', low=10, high=15),
        Integer(name='heatingTemp_day', low=15, high=30),
        Integer(name='CO2_pureCap', low=150, high=200),
        Integer(name='CO2_setpoint_night', low=400, high=800),
        Integer(name='CO2_setpoint_day', low=1199, high=1200),
        Integer(name='CO2_setpoint_lamp', low=800, high=1200),
        Integer(name='light_intensity', low=0, high=200),
        Integer(name='light_hours', low=10, high=20),
        Integer(name='light_endTime', low=18, high=24),
        Integer(name='light_maxIglob', low=299, high=300),
    ],
    'A4BST': [
        Categorical([38], name='num_days'),
        Categorical([10], name='heatingTemp_night'),
        Categorical([26], name='heatingTemp_day'),
        Categorical([182], name='CO2_pureCap'),
        Categorical([694], name='CO2_setpoint_night'),
        Categorical([1200], name='CO2_setpoint_day'),
        Categorical([912], name='CO2_setpoint_lamp'),
        Categorical([0], name='light_intensity'),
        Categorical([12], name='light_hours'),
        Categorical([24], name='light_endTime'),
        Categorical([300], name='light_maxIglob')
    ],
    'A5': [
        # Best Parameters: [38, 9.5, 26.1, 184, 603, 1199, 857, 4, 4.2, 6.0, 300]
        # Best NetProfit: 4.94
        Integer(name='num_days', low=35, high=40),
        Real(name='heatingTemp_night', low=8, high=12),
        Real(name='heatingTemp_day', low=24, high=28),
        Integer(name='CO2_pureCap', low=175, high=200),
        Integer(name='CO2_setpoint_night', low=600, high=800),
        Integer(name='CO2_setpoint_day', low=1199, high=1200),
        Integer(name='CO2_setpoint_lamp', low=800, high=1000),
        Integer(name='light_intensity', low=0, high=200),
        Real(name='light_hours', low=0, high=24),
        Real(name='light_endTime', low=0, high=24),
        Integer(name='light_maxIglob', low=299, high=300),
    ],
    'A6': [
        # Best Parameters: [193, 672, 1124, 949, 0, 8.0, 0.4]
        # Best NetProfit: 4.901
        # num_days=38,
        # heatingTemp_night=10,
        # heatingTemp_day=26,
        Integer(name='CO2_pureCap', low=100, high=200),
        Integer(name='CO2_setpoint_night', low=600, high=800),
        Integer(name='CO2_setpoint_day', low=800, high=1200),
        Integer(name='CO2_setpoint_lamp', low=800, high=1200),
        Integer(name='light_intensity', low=0, high=200),
        Real(name='light_hours', low=0, high=24),
        Real(name='light_endTime', low=0, high=24),
    ],
    'A7': [
        Integer(name='num_days', low=35, high=45),
        Integer(name='heatingTemp_night', low=5, high=15),
        Integer(name='heatingTemp_day', low=15, high=30),
        Integer(name='CO2_pureCap', low=100, high=200),
        Integer(name='CO2_setpoint_night', low=400, high=800),
        Integer(name='CO2_setpoint_day', low=800, high=1200),
        Integer(name='CO2_setpoint_lamp', low=800, high=1200),
        Integer(name='light_intensity', low=0, high=200),
        Integer(name='light_hours', low=0, high=24),
        Integer(name='light_endTime', low=0, high=24),
        Integer(name='light_maxIglob', low=200, high=300),
    ],
    'B1': [
        # D=39_TN=9.0_TD=19.0_CO2Cap=157.0_CO2N=562.0_CO2D=1129.0_CO2L=1128.0_LI=19.0_LH=5.0_LET=20.0_LMI=384.0.json
        Integer(name='num_days', low=35, high=45),
        Integer(name='heatingTemp_night', low=5, high=15),
        Integer(name='heatingTemp_day', low=15, high=30),
        Integer(name='CO2_pureCap', low=100, high=200),
        Integer(name='CO2_setpoint_night', low=400, high=800),
        Integer(name='CO2_setpoint_day', low=800, high=1200),
        Integer(name='CO2_setpoint_lamp', low=800, high=1200),
        Integer(name='light_intensity', low=0, high=200),
        Integer(name='light_hours', low=0, high=20),
        Integer(name='light_maxIglob', low=200, high=400),
    ],
    'B2': [
        # Best Parameters on B: [42, 5, 20, 184, 488, 955, 990, 64, 6, 267]
        # Best NetProfit on B: 4.714
        Integer(name='num_days', low=38, high=42),
        Integer(name='heatingTemp_night', low=5, high=10),
        Integer(name='heatingTemp_day', low=15, high=20),
        Integer(name='CO2_pureCap', low=150, high=250),
        Integer(name='CO2_setpoint_night', low=400, high=500),
        Integer(name='CO2_setpoint_day', low=800, high=1000),
        Integer(name='CO2_setpoint_lamp', low=800, high=1000),
        Integer(name='light_intensity', low=0, high=100),
        Integer(name='light_hours', low=4, high=6),
        Integer(name='light_maxIglob', low=100, high=300),
    ],
    'B3': [
        # 'x*': [43,
        # 5.971518146968884,
        # 21.97330475346169,
        # 193.14198622644798,
        # 455.34083842135925,
        # 998.8780689530633,
        # 1011.9683308027328,
        # 98.92770840263574,
        # 6.814450817359264,
        # 274.0868943754343],
        # 'y*': -2.052, on sim B
        Integer(name='num_days', low=41, high=43),
        Real(name='heatingTemp_night', low=4, high=6),
        Real(name='heatingTemp_day', low=18, high=22),
        Real(name='CO2_pureCap', low=180, high=200),
        Real(name='CO2_setpoint_night', low=450, high=500),
        Real(name='CO2_setpoint_day', low=900, high=1000),
        Real(name='CO2_setpoint_lamp', low=950, high=1050),
        Real(name='light_intensity', low=50, high=100),
        Real(name='light_hours', low=5, high=7),
        Real(name='light_maxIglob', low=250, high=300),
    ],
    'B4': [
        # 'x*': [42,
        # 13.234358952995306,
        # 29.75105761810473,
        # 184,
        # 773,
        # 943,
        # 1198,
        # 66,
        # 8.408759459289103,
        # 169],
        # 'y*': -2.209, on sim B
        Integer(name='num_days', low=35, high=45),
        Real(name='heatingTemp_night', low=0, high=15),
        Real(name='heatingTemp_day', low=15, high=30),
        Integer(name='CO2_pureCap', low=100, high=300),
        Integer(name='CO2_setpoint_night', low=400, high=800),
        Integer(name='CO2_setpoint_day', low=800, high=1200),
        Integer(name='CO2_setpoint_lamp', low=800, high=1200),
        Integer(name='light_intensity', low=0, high=200),
        Real(name='light_hours', low=0, high=20),
        Integer(name='light_maxIglob', low=100, high=300),
    ],
    'C1': [
        Integer(name='num_days', low=37, high=39),
        Integer(name='heatingTemp_night', low=8, high=12),
        Integer(name='heatingTemp_day', low=15, high=23),
        Integer(name='CO2_pureCap', low=145, high=180),
        Integer(name='CO2_setpoint_night', low=600, high=700),
        Integer(name='CO2_setpoint_day', low=1195, high=1200),
        Integer(name='CO2_setpoint_lamp', low=1000, high=1150),
        Integer(name='light_intensity', low=0, high=10),
        Integer(name='light_hours', low=10, high=18),
        Integer(name='light_endTime', low=17, high=22),
        Integer(name='light_maxIglob', low=299, high=310),
    ],
    'C2': [
        Integer(name='num_days', low=37, high=40),
        Integer(name='heatingTemp_night', low=9, high=11),
        Integer(name='heatingTemp_day', low=16, high=21),
        Integer(name='CO2_pureCap', low=148, high=170),
        Integer(name='CO2_setpoint_night', low=615, high=680),
        Integer(name='CO2_setpoint_day', low=1195, high=1200),
        Integer(name='CO2_setpoint_lamp', low=1000, high=1150),
        Integer(name='light_intensity', low=0, high=50),
        Integer(name='light_hours', low=10, high=15),
        Integer(name='light_endTime', low=17, high=21),
        Integer(name='light_maxIglob', low=295, high=305),
    ],
    'C2BST': [
        Categorical([40], name='num_days'),
        Categorical([9], name='heatingTemp_night'),
        Categorical([16], name='heatingTemp_day'),
        Categorical([159], name='CO2_pureCap'),
        Categorical([620], name='CO2_setpoint_night'),
        Categorical([1200], name='CO2_setpoint_day'),
        Categorical([1119], name='CO2_setpoint_lamp'),
        Categorical([0], name='light_intensity'),
        Categorical([10], name='light_hours'),
        Categorical([20], name='light_endTime'),
        Categorical([299], name='light_maxIglob')
    ],
}

BEST_PROFIT = np.inf
BEST_PARM = None
TIMEZONE = pytz.timezone('Europe/Amsterdam')
END_TIME = datetime(year=2021, month=7, day=13, hour=14, minute=55, tzinfo=TIMEZONE)

def make_day_scheme(dt1, v1, dt2, v2, dt3, v3, dt4, v4):
    return {
        f'r-{dt1}': v1,
        f'r+{dt2}': v2, 
        f's-{dt3}': v3,
        f's+{dt4}': v4,
    }


def get_func_and_callback(args, dim):

    LIGHT_END_TIME = 20
    
    @use_named_args(dimensions=dim)
    def netprofit(
        num_days,
        heatingTemp_night,
        heatingTemp_day,
        CO2_pureCap,
        CO2_setpoint_night,
        CO2_setpoint_day,
        CO2_setpoint_lamp,
        light_intensity,
        light_hours,
        light_endTime,
        light_maxIglob,
        plantDensity
    ):
        num_days = int(num_days)
        heatingTemp_night = round(float(heatingTemp_night), args.float_precision)
        heatingTemp_day = round(float(heatingTemp_day), args.float_precision)
        CO2_pureCap = round(float(CO2_pureCap), args.float_precision)
        CO2_setpoint_night = round(float(CO2_setpoint_night), args.float_precision)
        CO2_setpoint_day = round(float(CO2_setpoint_day), args.float_precision)
        CO2_setpoint_lamp = round(float(CO2_setpoint_lamp), args.float_precision)
        light_intensity = round(float(light_intensity), args.float_precision)
        light_hours = round(float(light_hours), args.float_precision)
        light_maxIglob = round(float(light_maxIglob), args.float_precision)
        light_endTime = round(float(light_endTime), args.float_precision)

        CP = ControlParamSimple()
        CP.set_endDate(num_days)
        heating_temp_scheme = {
            "01-01": {
                "r-1": heatingTemp_night,
                "r+1": heatingTemp_day, 
                "s-1": heatingTemp_day, 
                "s+1": heatingTemp_night
            }
        }
        CP.set_value("comp1.setpoints.temp.@heatingTemp", heating_temp_scheme)
        CP.set_value("common.CO2dosing.@pureCO2cap", CO2_pureCap)
        CO2_setpoint_scheme = {
            "01-01": {
                "r-1": CO2_setpoint_night, 
                "r+1": CO2_setpoint_day,
                "s-1": CO2_setpoint_day, 
                "s+1": CO2_setpoint_night
            }
        }
        CP.set_value("comp1.setpoints.CO2.@setpoint", CO2_setpoint_scheme)
        CP.set_value("comp1.setpoints.CO2.@setpIfLamps", CO2_setpoint_lamp)
        CP.set_value("comp1.illumination.lmp1.@enabled", light_intensity > 0)
        CP.set_value("comp1.illumination.lmp1.@intensity", light_intensity)
        CP.set_value("comp1.illumination.lmp1.@hoursLight", light_hours)
        CP.set_value("comp1.illumination.lmp1.@endTime", light_endTime)
        CP.set_value("comp1.illumination.lmp1.@maxIglob", light_maxIglob)
        # important need to search
        # CP.set_value("crp_lettuce.Intkam.management.@plantDensity", "1 90; 7 60; 14 40; 21 30; 28 20; 34 15")
        CP.set_value("crp_lettuce.Intkam.management.@plantDensity", str(plantDensity))


        control_name = f'D={num_days}_TN={heatingTemp_night}_TD={heatingTemp_day}_CO2Cap={CO2_pureCap}_CO2N={CO2_setpoint_night}_CO2D={CO2_setpoint_day}_CO2L={CO2_setpoint_lamp}_LI={light_intensity}_LH={light_hours}_LET={light_endTime}_LMI={light_maxIglob}_PD={plantDensity}.json'
        control_dir = f'{args.data_dir}/controls'
        output_dir = f'{args.data_dir}/outputs'
        CP.dump_json(control_dir, control_name)

        output = try_on_simulator(control_name, control_dir, output_dir, args.simulator)

        global BEST_PROFIT, BEST_PARM
        if - output['stats']['economics']['balance'] < BEST_PROFIT:
            BEST_PROFIT = - output['stats']['economics']['balance']
            files = glob.glob(f'{args.data_dir}/best_control/*')
            for f in files:
                os.remove(f)
            CP.dump_json(f'{args.data_dir}/best_control', f'BST={BEST_PROFIT}_{control_name}')
            BEST_PARM = control_name.split(".")[0]

        now = datetime.now(tz=TIMEZONE)
        if now > END_TIME: 
            try_on_simulator(f'BST={BEST_PROFIT}_{BEST_PARM}.json', f'{args.data_dir}/best_control', f'{args.data_dir}/best_output', args.simulator)
            sys.exit(0)

        return - output['stats']['economics']['balance']

    def save_result(res):
        result = {
            'space': res.space,
            'random_state': res.random_state,
            'xs': res.x_iters,
            'ys': list(res.func_vals),
            'x*': res.x,
            'y*': res.fun,
        }
        
        dump(result, f'{args.data_dir}/result.gz')
        
        with open(f'{args.data_dir}/result.log', 'w') as log_file:
            pprint(result, log_file)

    return netprofit, save_result

def add_plantDensity(dims):
    start_density_range = np.arange(80, 91, 5)  # 80,85,90
    end_density_range = np.arange(5, 16, 5)  # 5,10,15
    skip_day_range = np.arange(5, 11, 1)  # 5,6,7,8,9,10
    change_density_range = np.arange(5, 36, 5)  # 5,10,15,20,25,30,35

    max_days = dims[0].categories[0]
    control_densitys = []

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
            days = days+skip_day
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
            days = days + skip_days[i]
            density = density - change_densitys[i]
            control_density = f'{control_density}; {days} {density}'

        if density in end_density_range:
            control_densitys.append(control_density)
    
    dims_pd = copy.deepcopy(dims)
    dims_pd.append(Categorical(control_densitys, name='plantDensity'))
    return dims_pd

def add_plantDensity_fix(dims, pd):    
    dims_pd = copy.deepcopy(dims)
    dims_pd.append(Categorical([pd], name='plantDensity'))
    return dims_pd

def expand_best(best_no_pd):
    return [
        Integer(name='num_days', low=best_no_pd[0]-2, high=best_no_pd[0]+2),
        Integer(name='heatingTemp_night', low=best_no_pd[1]-3, high=best_no_pd[1]+3),
        Integer(name='heatingTemp_day', low=best_no_pd[2]-3, high=best_no_pd[2]+3),
        Integer(name='CO2_pureCap', low=best_no_pd[3]-15, high=best_no_pd[3]+15),
        Integer(name='CO2_setpoint_night', low=best_no_pd[4]-50, high=min(best_no_pd[4]+50, 1200)),
        Integer(name='CO2_setpoint_day', low=best_no_pd[5]-3, high=min(best_no_pd[5]+3, 1200)),
        Integer(name='CO2_setpoint_lamp', low=best_no_pd[6]-50, high=best_no_pd[6]+50),
        Integer(name='light_intensity', low=best_no_pd[7]-5, high=best_no_pd[7]+5),
        Integer(name='light_hours', low=best_no_pd[8]-2, high=best_no_pd[8]+2),
        Integer(name='light_endTime', low=best_no_pd[9]-2, high=best_no_pd[9]+2),
        Integer(name='light_maxIglob', low=best_no_pd[10]-5, high=best_no_pd[10]+5),
    ]

def optimize(args):
    opt = args.optimizer
    if opt == 'gp':
        opt_func = gp_minimize
    elif opt == 'forest':
        opt_func = forest_minimize
    elif opt == 'gbrt':
        opt_func = gbrt_minimize
    else:
        raise NotImplementedError(f'optimizer {opt} not supported!')

    best_pd = None
    best_no_pd = []

    for r in range(args.num_round):
        now = datetime.now(tz=NOW_TIMEZONE)
        if now > END_TIME: break
        
        if r == 0:
            dim = add_plantDensity(DIMS[args.start_point])
        elif r==1:
            dim = add_plantDensity_fix(DIMS[args.start_range], best_pd)
        else:
            dim = add_plantDensity_fix(expand_best(best_no_pd), best_pd)

        netprofit, save_result = get_func_and_callback(args, dim)

        one_round = opt_func(
            func=netprofit,
            dimensions=dim,
            n_initial_points=args.num_initial_points,
            n_calls=args.num_calls,
            random_state=args.random_seed,
            callback=save_result,
            verbose=args.logging
        )

        if r == 0: best_pd = one_round.x[-1]
        if r > 0: best_no_pd = one_round.x[0:-1]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-NR', '--num-round', type=int, default=10)
    parser.add_argument('-NC', '--num-calls', type=int, default=100, help='num of calls in each round')
    parser.add_argument('-NI', '--num-initial-points', type=int, default=10, help='num of initial calls in each round')
    parser.add_argument('-SP', '--start-point', type=str, default='C2BST')
    parser.add_argument('-SR', '--start-range', type=str, default='C2')
    parser.add_argument('-S', '--simulator', type=str, default='C')
    parser.add_argument('-O', '--optimizer', type=str, default='gbrt')
    parser.add_argument('-RS', '--random-seed', type=int, default=None)
    parser.add_argument('-L', '--logging', action='store_true')
    parser.add_argument('-P', '--float-precision', type=int, default=0)
    parser.add_argument('-D', '--data-dir', type=str, default=None)
    args = parser.parse_args()
    print(args)

    assert args.num_calls >= args.num_initial_points

    optimize(args)
    
    print(f'Best Parameters:', BEST_PARM)
    print(f'Best NetProfit:', BEST_PROFIT)
