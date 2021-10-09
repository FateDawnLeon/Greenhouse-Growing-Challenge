from constant import DEVICE
import os
import json
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
plt.ioff()


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

        x_min_idx = np.argmin(y)
        x_min = x[x_min_idx]
        y_min = y[x_min_idx]

        plt.scatter(x_min, y_min, marker='v')
        plt.plot(x, y, label=f"{split}-loss_min[{y_min:.4f}]@step[{x_min}]")

    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend()
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


if __name__ == '__main__':
    for i in range(4):
        model_dir = f"trained_models/ff_ensemble_B6OUACGJ_child[{i}]"
        loss_stats = load_json_data(f"{model_dir}/loss_stats.json")
        save_path = f"{model_dir}/loss_curve.png"
        plot_loss_curve(loss_stats, save_path)
