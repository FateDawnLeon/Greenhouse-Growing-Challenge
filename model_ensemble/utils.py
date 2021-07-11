import os
import json
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
plt.ioff()

from constant import DEVICE


def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    return data * std + mean


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(DEVICE)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def save_json_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def plot_loss_curve(loss_stats, save_path):
    plt.figure(dpi=300)

    for split in ['train', 'val']:
        step_losses = loss_stats[split]
        x = [p[0] for p in step_losses]
        y = [p[1] for p in step_losses]
        plt.plot(x, y, label=split)

        x_min = np.argmin(y)
        y_min = y[x_min]
        plt.scatter(x_min, y_min, marker='v')
        plt.annotate(f'loss[{y_min}]@step[{x_min+1}]', (x_min, y_min))

    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def get_ensemble_ckpt_paths(model_id="QTYW6TYM", step=50000):  # model_id should be something like QTYW6TYM
    ckpt_paths = []
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
    for name in os.listdir(f'{CURRENT_DIR}/trained_models'):
        isdir =  os.path.isdir(os.path.join(f'{CURRENT_DIR}/trained_models', name))
        if isdir and (model_id == 'all' or (model_id in name)):
            path = f'{CURRENT_DIR}/trained_models/{name}/checkpoints/step={step}.pth'
            ckpt_paths.append(path)
    return ckpt_paths
