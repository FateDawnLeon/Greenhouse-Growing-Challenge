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

def generate_control(config, mode, to_day):
    start_date = EARLY_START_DATE
    
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
    # lamp_type = str(config["lamp_type"])
    plantDensity = str(config["plantDensity"])
    scr1_ToutMax = float(config["scr1_ToutMax"])
    vent_startWnd = float(config["vent_startWnd"])

    light_endTime = {}
    light_hours = {}
    for d in range(to_day):
        cur = start_date + datetime.timedelta(days=d)
        key = "{:02d}-{:02d}".format(cur.day, cur.month)
        city = lookup("Amsterdam", database())
        sunrise, sunset = get_sun_rise_and_set(cur, city)
        light_endTime[key] = sunset
        light_hours[key] = sunset

    # light_endTime = 19.5
    # light_hours = 18
    CP = ControlParamSimple()
    CP.set_value("simset.@runMode", mode)
    CP.set_endDate(to_day)

    heating_temp_scheme = {}
    for d in range(to_day):
        cur = start_date + datetime.timedelta(days=d)
        key = "{:02d}-{:02d}".format(cur.day, cur.month)
        # city = lookup("Amsterdam", database())
        # starttime_a = get_sun_rise_and_set(cur, city)[0]
        # starttime_b = float(light_endTime) - float(light_hours)
        # starttime = min(starttime_a, starttime_b)
        # endtime = light_endTime
        # endtime = get_sun_rise_and_set(cur, city)[1]
        heating_temp_scheme[key] =  {
            str(0): heatingTemp_night,
            str(1): heatingTemp_day,
            str(light_endTime[key]-1): heatingTemp_day,
            str(light_endTime[key]): heatingTemp_night
        }
    # heating_temp_scheme = {
    #     "01-01": {
    #         str(light_endTime - light_hours): heatingTemp_night,
    #         str(light_endTime - light_hours+1): heatingTemp_day, 
    #         str(light_endTime-1): heatingTemp_day, 
    #         str(light_endTime): heatingTemp_night
    #     }
    # }

    CP.set_value("comp1.setpoints.temp.@heatingTemp", heating_temp_scheme)
    CP.set_value("common.CO2dosing.@pureCO2cap", CO2_pureCap)

    CO2_setpoint_scheme = {}
    for d in range(to_day):
        cur = start_date + datetime.timedelta(days=d)
        key = "{:02d}-{:02d}".format(cur.day, cur.month)
        # city = lookup("Amsterdam", database())
        # starttime_a = get_sun_rise_and_set(cur, city)[0]
        # starttime_b = float(light_endTime) - float(light_hours)
        # starttime = min(starttime_a, starttime_b)
        # endtime = light_endTime
        # endtime = get_sun_rise_and_set(cur, city)[1]
        CO2_setpoint_scheme[key] =  {
            str(0): CO2_setpoint_night,
            str(1): CO2_setpoint_day,
            str(light_endTime[key]-1): CO2_setpoint_day,
            str(light_endTime[key]): CO2_setpoint_night
        }
    # CO2_setpoint_scheme = {
    #     "01-01": {
    #         str(light_endTime - light_hours): CO2_setpoint_night,
    #         str(light_endTime - light_hours+1): CO2_setpoint_day, 
    #         str(light_endTime-1): CO2_setpoint_day, 
    #         str(light_endTime): CO2_setpoint_night
    #     }
    # }
            
    CP.set_value("comp1.setpoints.CO2.@setpoint", CO2_setpoint_scheme)
    CP.set_value("comp1.setpoints.CO2.@setpIfLamps", CO2_setpoint_lamp)
    CP.set_value("comp1.illumination.lmp1.@enabled", light_intensity > 0)
    CP.set_value("comp1.illumination.lmp1.@intensity", light_intensity)
    CP.set_value("comp1.illumination.lmp1.@hoursLight", light_hours)
    CP.set_value("comp1.illumination.lmp1.@endTime", light_endTime)
    CP.set_value("comp1.illumination.lmp1.@maxIglob", light_maxIglob)
    # CP.set_value("comp1.illumination.lmp1.@@type", lamp_type)
    CP.set_value("crp_lettuce.Intkam.management.@plantDensity", plantDensity)
    CP.set_value("comp1.screens.scr1.@ToutMax", scr1_ToutMax)
    CP.set_value("comp1.setpoints.ventilation.@startWnd", vent_startWnd)
    return CP.data

def objective(config, mode='pause', to_day=None, bo=True):

    if bo:
        to_day = int(config["duration"])
    else:
        to_day = to_day

    control = generate_control(config, mode, to_day)
    output = query_simulator(control, sim_id=SIM_ID)

    if bo:
        save_json_data(control, 'control.json')
        save_json_data(output, 'output.json')
    elif mode == 'pause':
        control_file_list = glob.glob(f'control_*.json')
        for f in control_file_list:
            os.remove(f)
        output_file_list = glob.glob(f'output_*.json')
        for f in output_file_list:
            os.remove(f)
        save_json_data(control, 'control_0.json')
        save_json_data(output, 'output_0.json')
    elif mode == 'step':
        control_file_list = glob.glob(f'control_*.json')
        call_step_time = max([int(f.split('_')[-1].split('.')[0]) for f in control_file_list])
        save_json_data(control, f'control_{call_step_time+1}.json')
        save_json_data(output, f'output_{call_step_time+1}.json')
    
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
        if len(data) > 0:
            print(key, data[-1])
        else:
            print(key, data)
        
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

    if bo:
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
