import os
import json
import datetime
from constant import START_DATE, SAMPLE_CONTROL_JSON_PATH


def valseq_to_scheme(vals, start_date):
    assert isinstance(vals, list) and len(vals) > 0 and len(vals) % 24 == 0

    scheme = {}
    for i in range(len(vals) // 24):
        date = start_date + datetime.timedelta(days=i)
        vals_day = vals[i*24:(i+1)*24]
        scheme[f'{date.day:02d}-{date.month:02d}'] = {str(t): vals_day[t] for t in range(24)}
    
    return scheme


class ControlParamSimple(object):
    def __init__(self, init_json_path=SAMPLE_CONTROL_JSON_PATH, start_date=START_DATE):
        super().__init__()
        with open(init_json_path, 'r') as f:
            self.data = json.load(f)

        assert type(start_date) == datetime.date
        self.start_date = start_date

    def dump_json(self, save_dir, save_name=None):
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = f'{hex(hash(self))}.json'
        with open(os.path.join(save_dir, save_name), 'w') as f:
            json.dump(self.data, f, indent=4)
    
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
    
    def set_endDate(self, num_days:int):
        end_date = self.start_date + datetime.timedelta(days=num_days)
        self.set_value("simset.@endDate", end_date.isoformat())

    def __repr__(self):
        return json.dumps(self.data, indent=4)

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return str(self.data).__hash__()
    