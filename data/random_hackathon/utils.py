import os
import json
import datetime
import requests


KEYS = {
    'hack': 'H17-KyEO-iDtD-mVGR'
}
URL = 'https://www.digigreenhouse.wur.nl/Kasprobeta/'


SAMPLE_CONTROL_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CONTROL_FILE = f'{SAMPLE_CONTROL_DIR}/sampleJson.json'


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


def valseq_to_scheme(vals, start_date):
    assert isinstance(vals, list) and len(vals) > 0 and len(vals) % 24 == 0

    scheme = {}
    for i in range(len(vals) // 24):
        date = start_date + datetime.timedelta(days=i)
        vals_day = vals[i*24:(i+1)*24]
        scheme[f'{date.day:02d}-{date.month:02d}'] = {str(t): vals_day[t] for t in range(24)}
    
    return scheme


class ControlParams(object):
    def __init__(self, start_date):
        super().__init__()
        self.start_date = start_date
        self.data = load_json_data(SAMPLE_CONTROL_FILE)

    def dump_json(self, save_dir, save_name=None):
        os.makedirs(save_dir, exist_ok=True)
        if not save_name:
            save_name = f'{hex(hash(self))}.json'
        save_json_data(self.data, os.path.join(save_dir, save_name))
    
    def set_value(self, key_path, value):
        keys = key_path.split('.')
        field = self.data
        for key in keys[:-1]:
            field = field[key]

        if type(value) in [int, float, bool, str, dict]:
            field[keys[-1]] = value
        elif type(value) == list:
            field[keys[-1]] = valseq_to_scheme(value, self.start_date)
        else:
            raise ValueError(f'value type of {value} not supported!')

    def __repr__(self):
        return json.dumps(self.data, indent=4)

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return str(self.data).__hash__()
