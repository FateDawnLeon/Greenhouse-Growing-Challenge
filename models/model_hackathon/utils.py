import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from shutil import copyfile
matplotlib.use('agg')
plt.ioff()


def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    return data * std + mean


def normalize_zero2one(arr, range):
    low, high = range
    return (arr - low) / (high - low + 1e-8)


def unnormalize_zero2one(arr, range):
    low, high = range
    return arr * (high - low) + low


def make_tensor(arr):
    return torch.from_numpy(arr).float().unsqueeze(0)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float()


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def save_json_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def plot_loss_curve(loss_stats, save_path, log_scale=False):
    plt.figure(dpi=300)

    for split in ['train', 'val']:
        step_losses = loss_stats[split]
        x = [p[0] for p in step_losses]
        y = [p[1] for p in step_losses]

        x_min_idx = np.argmin(y)
        x_min = x[x_min_idx]
        y_min = y[x_min_idx]

        plt.scatter(x_min, y_min, marker='v')
        plt.plot(x, y, label=f"{split}-loss_min[{y_min:.4f}]@step[{x_min}]")

    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend()
    if log_scale:
        plt.yscale('log')
    plt.savefig(save_path)
    plt.close()


# model_id should be something like QTYW6TYM
def get_ensemble_ckpt_paths(model_paths, model_id="QTYW6TYM", step=50000):
    ckpt_paths = []
    for name in os.listdir(model_paths):
        isdir = os.path.isdir(os.path.join(model_paths, name))
        if isdir and (model_id == 'all' or (model_id in name)):
            path = f'{model_paths}/{name}/checkpoints/step={step}.pth'
            ckpt_paths.append(path)
    return ckpt_paths


def organize_data(bo_result_dir, save_dir):
    names = os.listdir(bo_result_dir)
    folders = [name for name in names if name.startswith("objective")]

    os.makedirs(f"{save_dir}/controls", exist_ok=True)
    os.makedirs(f"{save_dir}/outputs", exist_ok=True)

    for folder in folders:
        control_path = os.path.join(bo_result_dir, folder, "control.json")
        output_path = os.path.join(bo_result_dir, folder, "output.json")
        data_id = folder.split('_')[1]

        copyfile(control_path, f"{save_dir}/controls/{data_id}.json")
        copyfile(output_path, f"{save_dir}/outputs/{data_id}.json")
        print(f"result {data_id} copied to {save_dir}...")


def get_output_data_dict(output, keys):
    return {key: output['data'][key]['data'] for key in keys}


def dict_to_dataframe(dict, dtype='float'):
    return pd.DataFrame(dict, dtype=dtype)


if __name__ == '__main__':
    for i in range(4):
        model_dir = f"trained_models/ff_ensemble_B6OUACGJ_child[{i}]"
        loss_stats = load_json_data(f"{model_dir}/loss_stats.json")
        save_path = f"{model_dir}/loss_curve.png"
        plot_loss_curve(loss_stats, save_path)
