import os
import json
import torch
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
        json.dump(data, f)


def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
