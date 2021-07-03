import os
import json
import requests
import numpy as np
from utils import ControlParamSimple


KEYS = {
    'A': 'C48A-ZRJQ-3wcq-rGuC-mEme',
    'B': 'C48B-PTmQ-89Kx-jqV5-3zRL' 
}
URL = 'https://www.digigreenhouse.wur.nl/Kasprobeta/model.aspx'


ENV_KEYS = [
    # related to greenhouse-inside environment parameters
    'common.Iglob.Value',
    'common.TOut.Value',
    'common.RHOut.Value',
    'common.Windsp.Value',

    # related to variable cost
    'common.Economics.PeakHour',
]


def get_output(control, sim_id):
    data = {"key": KEYS[sim_id], "parameters": json.dumps(control)}
    headers = {'ContentType': 'application/json'}

    while True:
        response = requests.post(URL, data=data, headers=headers, timeout=300)
        output = response.json()
        print(response, output['responsemsg'])

        if output['responsemsg'] == 'ok':
            break
        elif output['responsemsg'] == 'busy':
            continue
        else:
            raise ValueError('response message not expected!')
    
    return output


def get_EP(sim_id, num_days=50):
    CP = ControlParamSimple()
    CP.set_endDate(num_days=num_days)

    output = get_output(CP.data, sim_id)

    env_vals = []
    for key in ENV_KEYS:
        val = output['data'][key]['data']
        env_vals.append(val)
    env_vals = np.array(env_vals)
    env_vals = env_vals.T # T x N1

    print(env_vals.shape)

    return env_vals


def try_on_simulator(json_name, control_json_dir, output_json_dir, sim_id):
    with open(f'{control_json_dir}/{json_name}', 'r') as f:
        control = json.load(f)

    output = get_output(control, sim_id)

    os.makedirs(output_json_dir, exist_ok=True)
    with open(f'{output_json_dir}/{json_name}', 'w') as f:
        json.dump(output, f)
    
    print(json_name, 'finished')

    return output


if __name__ == '__main__':
    import argparse
    import threadpool

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num-trials', type=int, default=0)
    parser.add_argument('-T', '--num-workers', type=int, default=1)
    parser.add_argument('-S', '--simulator', choices=['A', 'B', 'C', 'D'], type=str, default='A')
    parser.add_argument('-C', '--clear-invalid-output', action='store_true', default=False)
    parser.add_argument('-F', '--control-json-file', type=str, default=None)
    parser.add_argument('-D', '--control-json-dir', type=str, default='control_jsons')
    parser.add_argument('-EP', '--get-ep', action='store_true')
    args = parser.parse_args()

    if args.get_ep:
        ep = get_EP(args.simulator)
        os.makedirs('common', exist_ok=True)
        np.save(f'common/EP-SIM={args.simulator}.npy', ep)
        exit(0)

    if args.control_json_file:
        output_json_dir = f'output_jsons_{args.simulator}'
    else:
        suffix = args.control_json_dir.split('_')[-1]
        output_json_dir = f'output_jsons_{suffix}_{args.simulator}'
    os.makedirs(output_json_dir, exist_ok=True)
    
    if args.clear_invalid_output:
        for name in os.listdir(output_json_dir):
            path = os.path.join(output_json_dir, name)
            file_size = os.path.getsize(path) / 1024
            if file_size < 5:
                os.remove(path)
                print(f'{path} has been removed.')

    if args.control_json_file:
        control_json_dir, file_name = os.path.split(args.control_json_file)
        control_json_names = [file_name]
        args.num_trials = 1
    else:
        # only test those control jsons that are not uploaded before
        control_json_dir = args.control_json_dir
        control_json_names = os.listdir(control_json_dir)
        output_json_names = os.listdir(output_json_dir)
        control_json_names = set(control_json_names).difference(output_json_names)

    # if jsons are not enough, just upload all valid ones	
    num_trials = min(args.num_trials, len(control_json_names))

    if num_trials > 0:
        control_json_names = list(control_json_names)[:num_trials]
        
        # using thread pool to automatically send concurrent requests
        pool = threadpool.ThreadPool(args.num_workers)
        func_vars = [([name, control_json_dir, output_json_dir, args.simulator], None) for name in control_json_names]
        tasks = threadpool.makeRequests(try_on_simulator, func_vars)

        for task in tasks:
            pool.putRequest(task)

        pool.wait()

    print(f'expected new trials: {args.num_trials}, actual new trials: {num_trials}')
