import json

import numpy as np
from tqdm import trange

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from utils import *


def preprocess_screen_threshold(val):
    if isinstance(val, str):
        val = [[float(x) for x in s.split()] for s in val.split(';')]
        val = list(np.array(val).flatten())
    else:
        # TODO: find a better value for screen threshold control
        val = [0, val, 10, val]
    return val


def preprocess_screen_material(val):
    return [val == 'scr_Transparent.par', val == 'scr_Shade.par', val == 'scr_Blackout.par']


class AGCDataset(Dataset):
    control_dir = './collect_data/control_jsons/'
    output_dir = './collect_data/output_jsons/'

    preproc_params_path = './collect_data/preproc_params.npz'

    control_keys = [
        ('common', 'CO2dosing', '@pureCO2cap'),

        ('comp1', 'heatingpipes', 'pipe1', '@maxTemp'),
        ('comp1', 'heatingpipes', 'pipe1', '@minTemp'),

        ('comp1', 'illumination', 'lmp1', '@enabled'),
        ('comp1', 'illumination', 'lmp1', '@endTime'),
        ('comp1', 'illumination', 'lmp1', '@hoursLight'),
        ('comp1', 'illumination', 'lmp1', '@intensity'),
        ('comp1', 'illumination', 'lmp1', '@maxIglob'),
        ('comp1', 'illumination', 'lmp1', '@maxPARsum'),

        ('comp1', 'screens', 'scr1', '@ToutMax'),
        ('comp1', 'screens', 'scr1', '@closeAbove'),
        ('comp1', 'screens', 'scr1', '@closeBelow'),
        ('comp1', 'screens', 'scr1', '@enabled'),
        ('comp1', 'screens', 'scr1', '@lightPollutionPrevention'),
        ('comp1', 'screens', 'scr1', '@material'),

        ('comp1', 'screens', 'scr2', '@ToutMax'),
        ('comp1', 'screens', 'scr2', '@closeAbove'),
        ('comp1', 'screens', 'scr2', '@closeBelow'),
        ('comp1', 'screens', 'scr2', '@enabled'),
        ('comp1', 'screens', 'scr2', '@lightPollutionPrevention'),

        ('comp1', 'setpoints', 'CO2', '@setpIfLamps'),
        ('comp1', 'setpoints', 'CO2', '@setpoint'),

        ('comp1', 'setpoints', 'temp', '@heatingTemp', '01-01', '8.0'),

        ('comp1', 'setpoints', 'ventilation', '@startWnd'),
        ('comp1', 'setpoints', 'ventilation', '@winLeeMax'),
        ('comp1', 'setpoints', 'ventilation', '@winLeeMin'),
        ('comp1', 'setpoints', 'ventilation', '@winWndMax'),
        ('comp1', 'setpoints', 'ventilation', '@winWndMin'),
    ]

    control_preprocessing = {
        ('comp1', 'screens', 'scr1', '@closeBelow'): preprocess_screen_threshold,
        ('comp1', 'screens', 'scr2', '@closeBelow'): preprocess_screen_threshold,
        ('comp1', 'screens', 'scr1', '@material'): preprocess_screen_material,
        ('comp1', 'screens', 'scr2', '@material'): preprocess_screen_material,
    }

    # keys not included are:
    # stats, common.Economics.PeakHour, comp1.Lmp1.ElecUse,
    # comp1.Plant.PlantDensity (since it's included in control),
    # and those in TARGET_KEYS
    output_keys = [
        'comp1.Air.T',
        'comp1.Air.RH',
        'comp1.Air.ppm',
        'common.Iglob.Value',
        'common.TOut.Value',
        'common.RHOut.Value',
        'common.Windsp.Value',
        'comp1.PARsensor.Above',
        'comp1.TPipe1.Value',
        'comp1.ConPipes.TSupPipe1',
        'comp1.PConPipe1.Value',
        'comp1.ConWin.WinLee',
        'comp1.ConWin.WinWnd',
        'comp1.Setpoints.SpHeat',
        'comp1.Setpoints.SpVent',
        'comp1.Scr1.Pos',
        'comp1.Scr2.Pos',
        'comp1.McPureAir.Value',
        'comp1.Plant.headFW',
        'comp1.Plant.shootDryMatterContent',
        'comp1.Plant.fractionGroundCover',
        'comp1.Plant.plantProjection'
    ]

    # location to save / load the parsed data
    x_path = './collect_data/x.npy'
    y_path = './collect_data/y.npy'

    def __init__(self):
        self.num_features = len(self.control_keys) + len(self.output_keys) + 1

        self.filenames = [name for name in os.listdir(self.control_dir) if
                          os.path.isfile(os.path.join(self.control_dir, name))]
        # remove those without corresponding output files
        self.filenames = [name for name in self.filenames if os.path.isfile(os.path.join(self.output_dir, name))]
        # remove those with bad response
        filenames = []
        for name in self.filenames:
            with open(os.path.join(self.output_dir, name), 'r') as f:
                j = json.load(f)
                if j['responsecode'] == 0:
                    filenames.append(name)
        self.filenames = filenames

        # get preprocessing parameter for output
        if os.path.isfile(self.preproc_params_path):
            preproc_params: dict = np.load(self.preproc_params_path)
            self.preproc_mean = preproc_params['mean']
            self.preproc_std = preproc_params['std']
        # otherwise, compute and save
        else:
            print('computing preprocessing parameters for output...')
            hours_total = 0
            for i in trange(len(self)):
                hours_total += self.parse_output(self.filenames[i]).shape[0]
            output_all = np.zeros((hours_total, len(self.output_keys)))
            idx = 0
            for i in trange(len(self)):
                output = self.parse_output(self.filenames[i])
                output_all[idx:idx + output.shape[0]] = output
                idx += output.shape[0]
            self.preproc_mean = np.mean(output_all, axis=0)
            self.preproc_std = np.std(output_all, axis=0)
            verify_output_path(self.preproc_params_path)
            np.savez(self.preproc_params_path, mean=self.preproc_mean, std=self.preproc_std)

    def parse_output(self, name):
        with open(os.path.join(self.output_dir, name), 'r') as f:
            output_json = json.load(f)

        output = []
        for key in self.output_keys:
            lst = output_json['data'][key]['data']
            lst = [-1.0 if x == 'NaN' else x for x in lst]
            output.append(lst)
        output = np.array(output)
        output = output.T

        return output

    def parse_control(self, name, num_hours):
        num_days = int(num_hours / 24)

        with open(os.path.join(self.control_dir, name), 'r') as f:
            control_json = json.load(f)
        control = []
        for key_tuple in self.control_keys:
            # get value from json
            val = control_json
            for k in key_tuple:
                val = val[k]

            # preprocessing
            if key_tuple in self.control_preprocessing:
                val = self.control_preprocessing[key_tuple](val)

            if isinstance(val, list):
                control.extend(val)
            else:
                control.append(val)
        control = np.repeat(np.array(control)[np.newaxis], num_hours, axis=0)

        # add spacing info to control
        spacing_control = control_json['crp_lettuce']['Intkam']['management']['@plantDensity']
        spacing_control = [tuple(map(lambda s: int(s), entry.split())) for entry in spacing_control.split('; ')]
        spacing = np.zeros(num_days, dtype=np.int)
        for day, val in spacing_control:
            spacing[day - 1:] = val
        spacing = np.repeat(spacing, 24)[:, np.newaxis]
        control = np.hstack((control, spacing))

        if control.dtype not in (np.float, np.int):
            print(name)
            for lst in control:
                print(lst)

        return control

    def __getitem__(self, idx):
        name = self.filenames[idx]

        # parse output file
        output = self.parse_output(name)
        num_hours = output.shape[0]

        # parse control_file
        control = self.parse_control(name, num_hours)

        assert output.dtype in (np.float, np.int), (output.dtype, output)
        assert control.dtype in (np.float, np.int), (control.dtype, control)

        # convert to tensor
        xs = torch.tensor(control[:-1]).float()
        ys = torch.tensor(output[1:]).float()

        # normalize output
        ys = self.normalize_target(ys)

        return xs, ys

    def __len__(self):
        return len(self.filenames)

    def normalize_target(self, y):
        return (y - np.array(self.preproc_mean)) / np.array(self.preproc_std)

    def denormalize_target(self, y):
        return y * np.array(self.preproc_std) + np.array(self.preproc_mean)


def pad_batch(batch, padding_value):
    xs, ys = list(zip(*batch))
    xs = pad_sequence(xs, padding_value=padding_value)
    ys = pad_sequence(ys, padding_value=padding_value)
    return xs, ys


def agc_dataloader(batch_size, shuffle=True, drop_last=True, padding_value=-1.0):
    dataset = AGCDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      collate_fn=lambda batch: pad_batch(batch, padding_value=padding_value))
