import os
import glob

from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from astral.geocoder import lookup, database
import datetime

from submit import BEST_DIR, SIM_ID
from space import SPACES
from utils import query_simulator, get_sun_rise_and_set, save_json_data, \
    ControlParamSimple, EARLY_START_DATE
import datetime

def generate_control(config):
    start_date = EARLY_START_DATE
    duration = int(config["duration"])
    heatingTemp_night = float(config["heatingTemp_night"])
    heatingTemp_day = float(config["heatingTemp_day"])
    CO2_pureCap = float(config["CO2_pureCap"])
    CO2_setpoint_night = float(config["CO2_setpoint_night"])
    CO2_setpoint_day = float(config["CO2_setpoint_day"])
    CO2_setpoint_lamp = float(config["CO2_setpoint_lamp"])
    light_intensity = float(config["light_intensity"])
    # light_hours = float(config["light_hours"])
    # light_endTime = float(config["light_endTime"])
    light_maxIglob = float(config["light_maxIglob"])
    plantDensity = str(config["plantDensity"])
    scr1_ToutMax = float(config["scr1_ToutMax"])
    vent_startWnd = float(config["vent_startWnd"])

    duration, plantDensity = config['nd_pd']

    light_endTime = {}
    light_hours = {}
    for d in range(duration):
        cur = start_date + datetime.timedelta(days=d)
        key = "{:02d}-{:02d}".format(cur.day, cur.month)
        city = lookup("Amsterdam", database())
        sunrise, sunset = get_sun_rise_and_set(cur, city)
        light_endTime[key] = sunset
        light_hours[key] = sunset - sunrise

    CP = ControlParamSimple()
    CP.set_endDate(duration)

    # heating_temp_scheme = {}
    # for d in range(duration):
    #     cur = start_date + datetime.timedelta(days=d)
    #     key = "{:02d}-{:02d}".format(cur.day, cur.month)
    #     city = lookup("Amsterdam", database())
    #     starttime_a = get_sun_rise_and_set(cur, city)[0]
    #     starttime_b = float(light_endTime) - float(light_hours)
    #     starttime = min(starttime_a, starttime_b)
    #     endtime = light_endTime
    #     # endtime = get_sun_rise_and_set(cur, city)[1]
    #     heating_temp_scheme[key] =  {
    #         str(starttime): heatingTemp_night,
    #         str(starttime+1): heatingTemp_day,
    #         str(endtime-1): heatingTemp_day,
    #         str(endtime): heatingTemp_night
    #     }
    heating_temp_scheme = {
        "01-01": {
            "r": heatingTemp_night,
            "r+1": heatingTemp_day, 
            "s-1": heatingTemp_day, 
            "s": heatingTemp_night
        }
    }

    CP.set_value("comp1.setpoints.temp.@heatingTemp", heating_temp_scheme)
    CP.set_value("common.CO2dosing.@pureCO2cap", CO2_pureCap)

    # CO2_setpoint_scheme = {}
    # for d in range(duration):
    #     cur = start_date + datetime.timedelta(days=d)
    #     key = "{:02d}-{:02d}".format(cur.day, cur.month)
    #     city = lookup("Amsterdam", database())
    #     starttime_a = get_sun_rise_and_set(cur, city)[0]
    #     starttime_b = float(light_endTime) - float(light_hours)
    #     starttime = min(starttime_a, starttime_b)
    #     endtime = light_endTime
    #     # endtime = get_sun_rise_and_set(cur, city)[1]
    #     CO2_setpoint_scheme[key] =  {
    #         str(starttime): CO2_setpoint_night,
    #         str(starttime+1): CO2_setpoint_day,
    #         str(endtime-1): CO2_setpoint_day,
    #         str(endtime): CO2_setpoint_night
    #     }
    CO2_setpoint_scheme = {
        "01-01": {
            "r": CO2_setpoint_night,
            "r+1": CO2_setpoint_day, 
            "s-1": CO2_setpoint_day, 
            "s": CO2_setpoint_night
        }
    }
            
    CP.set_value("comp1.setpoints.CO2.@setpoint", CO2_setpoint_scheme)
    CP.set_value("comp1.setpoints.CO2.@setpIfLamps", CO2_setpoint_lamp)
    CP.set_value("comp1.illumination.lmp1.@enabled", light_intensity > 0)
    CP.set_value("comp1.illumination.lmp1.@intensity", light_intensity)
    CP.set_value("comp1.illumination.lmp1.@hoursLight", light_hours)
    CP.set_value("comp1.illumination.lmp1.@endTime", light_endTime)
    CP.set_value("comp1.illumination.lmp1.@maxIglob", light_maxIglob)
    CP.set_value("crp_lettuce.Intkam.management.@plantDensity", plantDensity)
    CP.set_value("comp1.screens.scr1.@ToutMax", scr1_ToutMax)
    CP.set_value("comp1.setpoints.ventilation.@startWnd", vent_startWnd)
    return CP.data


def objective(config, checkpoint_dir=None):
    control = generate_control(config)
    save_json_data(control, 'control.json')
    
    output = query_simulator(control, sim_id=SIM_ID)
    save_json_data(output, 'output.json')
    
    balance = output['stats']['economics']['balance']
    print(f'Netprofit={balance}, Config={config}')
    print(output['stats']['economics'])
    PL_KEYS = [
        "comp1.Plant.headFW",
        "comp1.Plant.fractionGroundCover",
        "comp1.Plant.shootDryMatterContent",
        "comp1.Plant.qualityLoss",
    ]  # 4
    for key in PL_KEYS:
        data = output['data'][key]['data']
        print(key, data[-1])
    

    best_control_file_list = glob.glob(f'{BEST_DIR}/best_control_*.json')
    if len(best_control_file_list) == 0:
        best_control_file = f'{BEST_DIR}/best_control_{balance}.json'
        save_json_data(control, best_control_file)
    else:
        best_prev = float(os.path.splitext(best_control_file_list[0])[0].split('_')[-1])
        if balance > best_prev:
            for f in best_control_file_list:
                os.remove(f)
            best_control_file = f'{BEST_DIR}/best_control_{balance}.json'
            save_json_data(control, best_control_file)

    tune.report(netprofit=balance)


def run_search(args):
    if args.run_first:
        current_best_params = [SPACES[config_id] for config_id in args.run_first]
    else:
        current_best_params = None

    if args.try_one:
        algo = None
        search_space = SPACES[args.try_one]
        num_samples = 1
        space_name = args.try_one
    else:
        algo = HyperOptSearch(points_to_evaluate=current_best_params)
        algo = ConcurrencyLimiter(algo, max_concurrent=1)
        search_space = SPACES[args.search_space]
        num_samples = args.num_samples
        space_name = args.search_space

    analysis = tune.run(
        objective,
        name=f'Max_NetProfit_Space={space_name}',
        search_alg=algo,
        metric="netprofit",
        mode="max",
        num_samples=num_samples,
        config=search_space,
        verbose=1,
        local_dir="./search_results"
    )

    print('best config:', analysis.best_config)
    print('best netprofit of this round:', analysis.best_result['netprofit'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-SP', '--search-space', type=str, default=None)
    parser.add_argument('-NS', '--num-samples', type=int, default=1)
    parser.add_argument('-RF', '--run-first', type=str, nargs='+', default=None)
    parser.add_argument('-TO', '--try-one', type=str, default=None)
    args = parser.parse_args()

    run_search(args)
