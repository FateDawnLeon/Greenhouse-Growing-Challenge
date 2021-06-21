import json
import numpy as np
import matplotlib.pyplot as plt


def plot_data(data_path, save_dir):
    with open(data_path, 'r') as f:
        output = json.load(f)

    data = output["data"]
    filename = os.path.basename(data_path)[:-5]
    save_dir = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)

    for param_name in data:
        plt.figure()
        
        unit = data[param_name]["unit"]
        values = np.asarray(data[param_name]["data"])
        plt.plot(list(range(1, len(values)+1)), values)

        plt.title(param_name)
        plt.ylabel(f'Values / {unit}')
        plt.xlabel('Time / Hours')
        plt.savefig(f'{save_dir}/{param_name}.png')
        plt.close()


def plot_multi_data(data_paths, save_dir):
    data_all = {}
    for data_path in data_paths:
        with open(data_path, 'r') as f:
            output = json.load(f)
        data = output["data"]
        filename = os.path.basename(data_path)[:-5]
        data_all[filename] = data

    filename = list(data_all.keys())[0]
    param_names = data_all[filename].keys()
    os.makedirs(save_dir, exist_ok=True)
    
    for param_name in param_names:
        plt.figure()
        
        for filename in data_all:
            unit = data_all[filename][param_name]["unit"]
            values = data_all[filename][param_name]["data"]
            plt.plot(list(range(1, len(values)+1)), values, label=filename)
        
        plt.legend()
        plt.title(param_name)
        plt.xlabel('Time(hour)')
        plt.ylabel(f'Value({unit})')
        plt.savefig(f'{save_dir}/{param_name}.png')
        plt.close()


if __name__ == '__main__':
    import os
    import random
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--num-data', type=int)
    parser.add_argument('-D', '--data-dir', type=str)
    parser.add_argument('-F', '--data-path', type=str)
    parser.add_argument('-S', '--save-dir', type=str)
    parser.add_argument('-A', '--all-in-one', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    if args.data_path:
        plot_data(args.data_path, args.save_dir)
    else:
        # filenames = os.listdir(args.data_dir)[:args.num_data]
        filenames = random.sample(os.listdir(args.data_dir), args.num_data)
        if args.all_in_one:
            data_paths = [os.path.join(args.data_dir, filename) for filename in filenames]
            plot_multi_data(data_paths, args.save_dir)
        else:
            for filename in filenames:
                data_path = os.path.join(args.data_dir, filename)
                plot_data(data_path, args.save_dir)
                print(f'data {data_path} plotted...')
