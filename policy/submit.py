import glob, os
from parameters import log_folder
import os
import json
import datetime
import requests

prefix = 'data/local/model_final/'
experiment_path = prefix+log_folder

control_list = glob.glob(f'{experiment_path}/actions_*.json')

KEYS = {
    'A': 'C48A-ZRJQ-3wcq-rGuC-mEme',
    'B': 'C48B-PTmQ-89Kx-jqV5-3zRL',
    'C': 'C48A-ZRJQ-3wcq-rGuC-mEme',
    'D': 'C48B-PTmQ-89Kx-jqV5-3zRL' 
}
URL = 'https://www.digigreenhouse.wur.nl/Kasprobeta/model.aspx'
START_DATE = datetime.date(2021, 3, 4)


def query_simulator(control, sim_id):
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


def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_json_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)



for f in control_list:

    control = load_json_data(f)
    output = query_simulator(control, sim_id='C')

    balance = output['stats']['economics']['balance']
    print('best netprofit of final submission:', balance)