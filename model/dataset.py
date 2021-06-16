import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class AGCDataSet(Dataset):
    control_dir = './get_data/control_jsons'
    control_pattern = '{}.json'
    output_dir = './get_data/output_jsons'
    output_pattern = 'output-cp={}.json.json'

    control_keys = [
        ('comp1', 'setpoints', 'temp', '@heatingTemp', '01-01', 'r+1'),
        ('comp1', 'setpoints', 'temp', '@heatingTemp', '01-01', 'r-1'),
        ('comp1', 'setpoints', 'temp', '@heatingTemp', '01-01', 's+1'),
        ('comp1', 'setpoints', 'temp', '@heatingTemp', '01-01', 's+1'),
        ('common', 'CO2dosing', '@pureCO2cap'),
        ('comp1', 'setpoints', 'CO2', '@setpoint'),
        ('comp1', 'illumination', 'lmp1', '@hoursLight'),
        ('comp1', 'illumination', 'lmp1', '@endTime'),
        ('comp1', 'setpoints', 'ventilation', '@startWnd'),
    ]

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
        'comp1.Plant.fractionGroundCover',
        'comp1.Plant.plantProjection'
    ]

    # values we want to predict
    target_keys = [
        'comp1.Plant.headFW',
        'comp1.Plant.shootDryMatterContent'
    ]

    # location to save / load the parsed data
    x_path = './get_data/x.npy'
    y_path = './get_data/y.npy'

    def __init__(self):
        self.num_features = len(self.control_keys) + len(self.output_keys) + 1
        self.num_target = len(self.target_keys)

        self.tokens = [os.path.splitext(f)[0] for f in os.listdir(self.control_dir) if
                       os.path.isfile(os.path.join(self.control_dir, f))]
        # remove those without corresponding output files
        self.tokens = [token for token in self.tokens if
                       os.path.isfile(os.path.join(self.output_dir, self.output_pattern.format(token)))]

    def __getitem__(self, idx):
        token = self.tokens[idx]

        # parse output file
        with open(os.path.join(self.output_dir, self.output_pattern.format(token)), 'r') as f:
            output_json = json.load(f)
        output = np.array([output_json['data'][key]['data'] for key in self.output_keys])
        output = output.T
        # output shape should be (days x 24, 20)

        num_hours = output.shape[0]
        num_days = int(num_hours / 24)

        # parse control file
        with open(os.path.join(self.control_dir, self.control_pattern.format(token)), 'r') as f:
            control_json = json.load(f)
        control = []
        for key_tuple in self.control_keys:
            val = control_json
            for k in key_tuple:
                val = val[k]
            control.append(val)
        control = np.repeat(np.array(control)[np.newaxis], num_hours, axis=0)
        # control shape should be (days x 24, 9)

        # add spacing info to control
        spacing_control = control_json['crp_lettuce']['Intkam']['management']['@plantDensity']
        spacing_control = [tuple(map(lambda s: int(s), entry.split())) for entry in spacing_control.split('; ')]
        spacing = np.zeros(num_days, dtype=np.int)
        for day, val in spacing_control:
            spacing[day - 1:] = val
        spacing = np.repeat(spacing, 24)[:, np.newaxis]
        control = np.hstack((control, spacing))

        # aggregate to final input features
        # note that the last input is cut since there's nothing to predict
        x = np.hstack((control[:-1], output[:-1]))

        # parse target values
        # note that the first target is cut since there's no corresponding input
        y = np.array([output_json['data'][key]['data'] for key in self.target_keys])
        y = y.T
        y = y[1:]

        # convert to Tensor
        x = torch.Tensor(x).float()
        y = torch.Tensor(y).float()

        # normalize target
        y = self.normalize_target(y)

        return x, y

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def normalize_target(y, mean=(0, 0), var=(300, 0.1)):
        return (y - np.array(mean)) / np.array(var)

    @staticmethod
    def denormalize_target(y, mean=(0, 0), var=(300, 0.1)):
        return y * np.array(var) + np.array(mean)
