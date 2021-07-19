import os
import json
import argparse
from shutil import copyfile


parser = argparse.ArgumentParser()
parser.add_argument('--output-data-dir', type=str, required=True)
parser.add_argument('--copy-file', action='store_true', default=False)
parser.add_argument('--save-dir', type=str, default=None)
parser.add_argument('--FW-min', type=float, default=None)
parser.add_argument('--FW-max', type=float, default=None)
parser.add_argument('--DMC-min', type=float, default=None)
parser.add_argument('--DMC-max', type=float, default=None)
parser.add_argument('--balance-min', type=float, default=None)
parser.add_argument('--balance-max', type=float, default=None)
parser.add_argument('--gains-min', type=float, default=None)
parser.add_argument('--gains-max', type=float, default=None)
args = parser.parse_args()


for filename in os.listdir(args.output_data_dir):
    selected = True
    filepath = os.path.join(args.output_data_dir, filename) 
    with open(filepath, 'r') as f:
        output = json.load(f)

    if 'stats' not in output:
        continue

    if (args.FW_min is not None) and (output['stats']['economics']['headFreshWeight'] < args.FW_min):
        selected = False
    if (args.FW_max is not None) and (output['stats']['economics']['headFreshWeight'] > args.FW_max):
        selected = False
    if (args.DMC_min is not None) and (output['stats']['economics']['bonusMalusOnDMC'] < args.DMC_min):
        selected = False
    if (args.DMC_max is not None) and (output['stats']['economics']['bonusMalusOnDMC'] > args.DMC_max):
        selected = False
    if (args.balance_min is not None) and (output['stats']['economics']['balance'] < args.balance_min):
        selected = False
    if (args.balance_max is not None) and (output['stats']['economics']['balance'] > args.balance_max):
        selected = False
    if (args.gains_min is not None) and (output['stats']['economics']['gains']['total'] < args.gains_min):
        selected = False
    if (args.gains_max is not None) and (output['stats']['economics']['gains']['total'] > args.gains_max):
        selected = False

    if selected:
        print(filename)
        if args.copy_file and args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            copyfile(filepath, os.path.join(args.save_dir, filename))
