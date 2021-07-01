import matplotlib
import matplotlib.pyplot as plt
import torch
from model import Model
from data import SupervisedModelDataset, zscore_normalize, compute_mean_std
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('agg')
plt.ioff()


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return zscore_normalize(data, self.mean, self.std)


class DataNormalizer(object):
    def __init__(self, cp_mean_std, ep_mean_std, op_mean_std):
        self.cp_normalize = Normalizer(*cp_mean_std)
        self.ep_normalize = Normalizer(*ep_mean_std)
        self.op_normalize = Normalizer(*op_mean_std)

    def __call__(self, cp, ep, op_pre, op_cur):
        cp = self.cp_normalize(cp)
        ep = self.ep_normalize(ep)
        op_pre = self.op_normalize(op_pre)
        op_cur = self.op_normalize(op_cur)
        return cp, ep, op_pre, op_cur


def plot_running_stats_step(stats_name, stats_data, save_path, title=None):
    plt.figure(dpi=300)
    plt.plot(stats_data)
    plt.title(title or f'step vs. {stats_name}')
    plt.xlabel('step')
    plt.ylabel(stats_name)
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('-B', '--batch-size', type=int, default=64)
    parser.add_argument('-E', '--max-epochs', type=int, default=1)
    parser.add_argument('-L', '--log-interval', type=int, default=100)
    parser.add_argument('-D', '--data-dirs', nargs='+', default=None)
    parser.add_argument('-R', '--root-dir', type=str, required=True)
    parser.add_argument('-S', '--save-interval', type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(f'{args.root_dir}/checkpoints', exist_ok=True)

    cp_mean_std, ep_mean_std, op_mean_std = compute_mean_std(args.data_dirs)
    normalize = DataNormalizer(cp_mean_std, ep_mean_std, op_mean_std)
    dataset = SupervisedModelDataset(args.data_dirs, transform=normalize)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    in_features = dataset.cp_dim + dataset.ep_dim + dataset.op_dim
    out_features = dataset.op_dim
    model = Model(in_features, out_features)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()

    loss_stats = []
    model.train()
    num_step = 0
    for epoch in range(1, args.max_epochs+1):
        for i, (cp, ep, op_pre, op_cur) in enumerate(dataloader):
            pred_op_cur = model(cp, ep, op_pre)
            loss = criterion(pred_op_cur, op_cur)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_stats.append(loss.item())
            num_step += 1

            if num_step % args.log_interval == 0:
                print(
                    f'Epoch[{epoch}/{args.max_epochs}] Batch[{i+1}/{len(dataloader)}] {criterion}: {loss.item():.4f}')
                plot_running_stats_step(
                    repr(criterion), loss_stats, f'{args.root_dir}/loss-curve.png')

            if num_step % args.save_interval == 0:
                torch.save(
                    model.state_dict(), f'{args.root_dir}/checkpoints/model_step={num_step}.pth')
