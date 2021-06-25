from skopt.utils import use_named_args
from skopt.space import Integer, Real
from skopt import gp_minimize
from utils import ControlParams
from run_simulation import try_on_simulator
import pickle
import warnings
warnings.filterwarnings('ignore')


dimensions = {
    'A': [
        Integer(name='duration', low=20, high=50),
        Real(name='temp_night', low=10, high=30),
        Real(name='temp_day', low=10, high=30),
        Real(name='CO2_supply_rate', low=100, high=200),
        Real(name='CO2_setpoint_night', low=400, high=1200),
        Real(name='CO2_setpoint_day', low=400, high=1200),
        Real(name='CO2_setpoint_lamp', low=400, high=1200),
        Real(name='light_intensity', low=50, high=100),
        Real(name='light_hours', low=0, high=24),
        Real(name='light_endTime', low=18, high=24),
        Real(name='light_maxIglob', low=100, high=200),
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
    ]
}


def get_func(args):

    @use_named_args(dimensions=dimensions[args.dimension_spec])
    def netprofit(duration,
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
        temp_night = float(temp_night)
        temp_day = float(temp_day)
        CO2_supply_rate = float(CO2_supply_rate)
        CO2_setpoint_night = float(CO2_setpoint_night)
        CO2_setpoint_day = float(CO2_setpoint_day)
        CO2_setpoint_lamp = float(CO2_setpoint_lamp)
        light_intensity = float(light_intensity)
        light_hours = float(light_hours)
        light_endTime = float(light_endTime)
        light_maxIglob = float(light_maxIglob)

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
            intensity=light_intensity,
            hoursLight=light_hours,
            endTime=light_endTime,
            maxIglob=light_maxIglob,
        )

        control_name = f'D={duration}_TN={temp_night}_TD={temp_day}_CO2Cap={CO2_supply_rate}_CO2N={CO2_setpoint_night}_CO2D={CO2_setpoint_day}_CO2L={CO2_setpoint_lamp}_LI={light_intensity}_LH={light_hours}_LET={light_endTime}_LMI={light_maxIglob}.json'
        control_dir = f'{args.data_dir}/controls'
        output_dir = f'{args.data_dir}/outputs'
        CP.save_as_json(control_dir, control_name)

        output = try_on_simulator(
            control_name, control_dir, output_dir, args.simulator)

        netprofit = output['stats']['economics']['balance']
        print('>>> netprofit:', netprofit)
        return - netprofit

    def save_result(res):
        with open(f'{args.data_dir}/result.pkl', 'wb') as output:
            pickle.dump({'inputs': res.x_iters, 'outputs': res.func_vals},
                        output, pickle.HIGHEST_PROTOCOL)

        with open(f'{args.data_dir}/log.txt', 'a') as f:
            print(f'>>> call steps:{len(res.x_iters)}', file=f)
            print(f'input values:{res.x_iters[-1]}', file=f)
            print(f'target values:{res.func_vals[-1]}', file=f)
            print(
                f'<<< best param: {res.x}, best netproft:{- res.fun}\n', file=f)

    return netprofit, save_result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num-calls', type=int, default=20)
    parser.add_argument('-I', '--num-initial-points', type=int, default=10)
    parser.add_argument('-R', '--random-seed', type=int, default=123)
    parser.add_argument('-S', '--dimension-spec', type=str, default='A')
    parser.add_argument('-D', '--data-dir', type=str, required=True)
    parser.add_argument('-SIM', '--simulator', type=str, default='A')
    args = parser.parse_args()
    print(args)

    assert args.num_calls >= args.num_initial_points

    target_func, callback_func = get_func(args)

    res = gp_minimize(
        func=target_func,
        dimensions=dimensions[args.dimension_spec],
        n_initial_points=args.num_initial_points,
        n_calls=args.num_calls,
        random_state=args.random_seed,
        callback=callback_func
    )
    print(f"Best parameters: {res.x}")
    print(f"Best netprofit: {- res.fun}")
