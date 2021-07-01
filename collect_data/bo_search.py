import warnings
warnings.filterwarnings('ignore')

from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.utils import use_named_args, dump
from skopt.space import Integer, Real, Categorical

from pprint import pprint
from utils import ControlParams, ControlParamSimple
from run_simulation import try_on_simulator


DIMS = {
    'A': [
        # runtime: method=gbrt_init=1000_ncall=10000_RS=42_SP=A_P=1_sim=A
        # Best Parameters: [37, 29.331173428460527, 19.740135016200067, 105.54579140908399, 735.4649817201819, 800.7794533665503, 1114.7327640161704, 10.831808389882115, 12.963267259805608, 16.377776700064487, 347.19371387181457]
        # Best NetProfit: 3.923
        Integer(name='duration', low=30, high=50),
        Real(name='temp_night', low=15, high=30),
        Real(name='temp_day', low=5, high=20),
        Real(name='CO2_supply_rate', low=100, high=200),
        Real(name='CO2_setpoint_night', low=400, high=800),
        Real(name='CO2_setpoint_day', low=800, high=1200),
        Real(name='CO2_setpoint_lamp', low=800, high=1200),
        Real(name='light_intensity', low=0, high=200),
        Real(name='light_hours', low=0, high=24),
        Real(name='light_endTime', low=0, high=24),
        Real(name='light_maxIglob', low=100, high=400),
    ],
    'B': [
        Integer(name='duration', low=35, high=45),
        Real(name='temp_night', low=10, high=30),
        Real(name='temp_day', low=10, high=20),
        Real(name='CO2_supply_rate', low=100, high=200),
        Real(name='CO2_setpoint_night', low=400, high=1200),
        Real(name='CO2_setpoint_day', low=400, high=1200),
        Real(name='CO2_setpoint_lamp', low=400, high=1200),
        Real(name='light_intensity', low=0, high=200),
        Real(name='light_hours', low=4, high=20),
        Real(name='light_endTime', low=0, high=24),
        Real(name='light_maxIglob', low=100, high=200),
    ],
    'C': [
        Integer(name='duration', low=30, high=50),
        Integer(name='temp_night', low=0, high=20),
        Integer(name='temp_day', low=10, high=30),
        Integer(name='CO2_supply_rate', low=100, high=200),
        Integer(name='CO2_setpoint_night', low=0, high=1200),
        Integer(name='CO2_setpoint_day', low=0, high=1200),
        Integer(name='CO2_setpoint_lamp', low=0, high=1200),
        Integer(name='light_intensity', low=0, high=200),
        Integer(name='light_hours', low=0, high=24),
        Integer(name='light_endTime', low=0, high=24),
        Integer(name='light_maxIglob', low=100, high=300),
    ],
    'D': [
        # Best Parameters: [38, 10, 26, 182, 694, 1200, 912, 0, 12, 24, 300]
        # Best NetProfit: 4.941
        Integer(name='duration', low=38, high=42),
        Integer(name='temp_night', low=10, high=15),
        Integer(name='temp_day', low=15, high=30),
        Integer(name='CO2_supply_rate', low=150, high=200),
        Integer(name='CO2_setpoint_night', low=400, high=800),
        Integer(name='CO2_setpoint_day', low=1199, high=1200),
        Integer(name='CO2_setpoint_lamp', low=800, high=1200),
        Integer(name='light_intensity', low=0, high=200),
        Integer(name='light_hours', low=10, high=20),
        Integer(name='light_endTime', low=18, high=24),
        Integer(name='light_maxIglob', low=299, high=300),
    ],
    'E': [
        # Best Parameters: [38, 9.5, 26.1, 184, 603, 1199, 857, 4, 4.2, 6.0, 300]
        # Best NetProfit: 4.94
        Integer(name='duration', low=35, high=40),
        Real(name='temp_night', low=8, high=12),
        Real(name='temp_day', low=24, high=28),
        Integer(name='CO2_supply_rate', low=175, high=200),
        Integer(name='CO2_setpoint_night', low=600, high=800),
        Integer(name='CO2_setpoint_day', low=1199, high=1200),
        Integer(name='CO2_setpoint_lamp', low=800, high=1000),
        Integer(name='light_intensity', low=0, high=200),
        Real(name='light_hours', low=0, high=24),
        Real(name='light_endTime', low=0, high=24),
        Integer(name='light_maxIglob', low=299, high=300),
    ],
    'F': [
        # Best Parameters: [193, 672, 1124, 949, 0, 8.0, 0.4]
        # Best NetProfit: 4.901
        Integer(name='CO2_supply_rate', low=100, high=200),
        Integer(name='CO2_setpoint_night', low=600, high=800),
        Integer(name='CO2_setpoint_day', low=800, high=1200),
        Integer(name='CO2_setpoint_lamp', low=800, high=1200),
        Integer(name='light_intensity', low=0, high=200),
        Real(name='light_hours', low=0, high=24),
        Real(name='light_endTime', low=0, high=24),
    ],
    'AA': [
        Integer(name='duration', low=35, high=45),
        Integer(name='temp_night', low=5, high=15),
        Integer(name='temp_day', low=15, high=30),
        Integer(name='CO2_supply_rate', low=100, high=200),
        Integer(name='CO2_setpoint_night', low=400, high=800),
        Integer(name='CO2_setpoint_day', low=800, high=1200),
        Integer(name='CO2_setpoint_lamp', low=800, high=1200),
        Integer(name='light_intensity', low=0, high=200),
        Integer(name='light_hours', low=0, high=24),
        Integer(name='light_endTime', low=0, high=24),
        Integer(name='light_maxIglob', low=200, high=300),
    ],
    'BB': [
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
        Integer(name='light_maxIglob', low=200, high=400),
        # Categorical(name='plant_density', categories=[
        #     "1 80; 10 40; 20 30; 25 20; 30 10",
        #     "1 90; 7 60; 14 40; 21 30; 28 20; 34 15",
        # ])
    ]
}


class NetProfitOptimizer(object):
    # deprecated: dont use this one
    def __init__(self, args):
        super().__init__()
        self.dimensions = DIMS[args.dimension_spec]
        self.data_dir = args.data_dir
        self.n_calls = args.num_calls
        self.n_initial_points = args.num_initial_points
        self.random_state = args.random_seed
        self.simulator = args.simulator
        self.float_precision = args.float_precision
        
        if args.dimension_spec == 'F':
            target_func = self.netprofit2
        else:
            target_func = self.netprofit
            
        self.target_func = use_named_args(dimensions=self.dimensions)(target_func)

    def netprofit(self,
                duration,
                temp_night,
                temp_day,
                CO2_supply_rate,
                CO2_setpoint_night,
                CO2_setpoint_day,
                CO2_setpoint_lamp,
                light_intensity,
                light_hours,
                light_endTime,
                light_maxIglob):
        
        duration = int(duration)
        temp_night = round(float(temp_night), self.float_precision)
        temp_day = round(float(temp_day), self.float_precision)
        CO2_supply_rate = round(float(CO2_supply_rate), self.float_precision)
        CO2_setpoint_night = round(float(CO2_setpoint_night), self.float_precision)
        CO2_setpoint_day = round(float(CO2_setpoint_day), self.float_precision)
        CO2_setpoint_lamp = round(float(CO2_setpoint_lamp), self.float_precision)
        light_intensity = round(float(light_intensity), self.float_precision)
        light_hours = round(float(light_hours), self.float_precision)
        light_endTime = round(float(light_endTime), self.float_precision)
        light_maxIglob = round(float(light_maxIglob), self.float_precision)

        CP = ControlParams()
        CP.set_end_date(duration=duration)
        CP.set_temperature(
            heatingTemp={"01-01": {"r-1": temp_night,
                                    "r+1": temp_day, "s-1": temp_day, "s+1": temp_night}}
        )
        CP.set_CO2(
            pureCO2cap=CO2_supply_rate,
            setpoint={"01-01": {"r-1": CO2_setpoint_night, "r+1": CO2_setpoint_day,
                                "s-1": CO2_setpoint_day, "s+1": CO2_setpoint_night}},
            setpIfLamps=CO2_setpoint_lamp
        )
        CP.set_illumination(
            enabled=True if light_intensity > 0 else False,
            intensity=light_intensity,
            hoursLight=light_hours,
            endTime=light_endTime,
            maxIglob=light_maxIglob,
        )

        control_name = f'D={duration}_TN={temp_night}_TD={temp_day}_CO2Cap={CO2_supply_rate}_CO2N={CO2_setpoint_night}_CO2D={CO2_setpoint_day}_CO2L={CO2_setpoint_lamp}_LI={light_intensity}_LH={light_hours}_LET={light_endTime}_LMI={light_maxIglob}.json'
        control_dir = f'{self.data_dir}/controls'
        output_dir = f'{self.data_dir}/outputs'
        CP.save_as_json(control_dir, control_name)

        output = try_on_simulator(control_name, control_dir, output_dir, self.simulator)

        return - output['stats']['economics']['balance']

    def netprofit2(self, 
                CO2_supply_rate,
                CO2_setpoint_night,
                CO2_setpoint_day,
                CO2_setpoint_lamp,
                light_intensity,
                light_hours,
                light_endTime):
        return self.netprofit(
            duration=38,
            temp_night=10,
            temp_day=26,
            CO2_supply_rate=CO2_supply_rate,
            CO2_setpoint_night=CO2_setpoint_night,
            CO2_setpoint_day=CO2_setpoint_day,
            CO2_setpoint_lamp=CO2_setpoint_lamp,
            light_intensity=light_intensity,
            light_hours=light_hours,
            light_endTime=light_endTime,
            light_maxIglob=300
        )

    def save_result(self, res):
        result = {
            'space': res.space,
            'random_state': res.random_state,
            'xs': res.x_iters,
            'ys': list(res.func_vals),
            'x*': res.x,
            'y*': res.fun,
        }
        
        dump(result, f'{self.data_dir}/result.gz')
        
        with open(f'{self.data_dir}/result.log', 'w') as log_file:
            pprint(result, log_file)

    def optimize(self, opt, verbose):
        if opt == 'gp':
            opt_func = gp_minimize
        elif opt == 'forest':
            opt_func = forest_minimize
        elif opt == 'gbrt':
            opt_func = gbrt_minimize
        else:
            raise NotImplementedError(f'optimizer {opt} not supported!')

        res = opt_func(
            func=self.target_func,
            dimensions=self.dimensions,
            n_initial_points=self.n_initial_points,
            n_calls=self.n_calls,
            random_state=self.random_state,
            callback=self.save_result,
            verbose=verbose
        )
        return res


def get_func_and_callback(args):
    
    @use_named_args(dimensions=DIMS[args.dimension_spec])
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
        light_maxIglob
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
        light_endTime = round(float(light_endTime), args.float_precision)
        light_maxIglob = round(float(light_maxIglob), args.float_precision)

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

        control_name = f'D={num_days}_TN={heatingTemp_night}_TD={heatingTemp_day}_CO2Cap={CO2_pureCap}_CO2N={CO2_setpoint_night}_CO2D={CO2_setpoint_day}_CO2L={CO2_setpoint_lamp}_LI={light_intensity}_LH={light_hours}_LET={light_endTime}_LMI={light_maxIglob}.json'
        control_dir = f'{args.data_dir}/controls'
        output_dir = f'{args.data_dir}/outputs'
        CP.dump_json(control_dir, control_name)

        output = try_on_simulator(control_name, control_dir, output_dir, args.simulator)

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

    netprofit, save_result = get_func_and_callback(args)

    res = opt_func(
        func=netprofit,
        dimensions=DIMS[args.dimension_spec],
        n_initial_points=args.num_initial_points,
        n_calls=args.num_calls,
        random_state=args.random_seed,
        callback=save_result,
        verbose=args.logging
    )
    return res


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-NC', '--num-calls', type=int, default=5)
    parser.add_argument('-NI', '--num-initial-points', type=int, default=1)
    parser.add_argument('-RS', '--random-seed', type=int, default=None)
    parser.add_argument('-DS', '--dimension-spec', type=str, default='A')
    parser.add_argument('-S', '--simulator', type=str, default='A')
    parser.add_argument('-D', '--data-dir', type=str, default=None)
    parser.add_argument('-O', '--optimizer', type=str, default='gp')
    parser.add_argument('-L', '--logging', action='store_true')
    parser.add_argument('-P', '--float-precision', type=int, default=0)
    args = parser.parse_args()
    print(args)

    assert args.num_calls >= args.num_initial_points

    # optimizer = NetProfitOptimizer(args)
    # res  = optimizer.optimize(opt=args.optimizer, verbose=args.logging)

    res = optimize(args)
    
    print(f'Best Parameters:', res.x)
    print(f'Best NetProfit:', - res.fun)
