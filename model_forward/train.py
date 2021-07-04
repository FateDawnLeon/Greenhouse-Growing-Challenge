import matplotlib
import matplotlib.pyplot as plt
import torch
from model import Model
from data import SupervisedModelDataset, zscore_normalize, compute_mean_std
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch import nn
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('agg')
plt.ioff()


def plot_running_stats_step(stats_name, stats_data, save_path, title=None):
    plt.figure(dpi=300)
    plt.plot(stats_data)
    plt.title(title or f'step vs. {stats_name}')
    plt.xlabel('step')
    plt.ylabel(stats_name)
    plt.savefig(save_path)
    plt.close()


def get_batch(dataloader):
    while True:
        yield from dataloader


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_step(model, iterloader, criterion, optimizer):
    cp, ep, op_pre, op_cur = next(iterloader)
        
    pred_op_cur = model(cp, ep, op_pre)
    loss = criterion(pred_op_cur, op_cur)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--max-steps', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data-dirs', nargs='+', required=True)
    parser.add_argument('--root-dir', type=str, required=True)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--scheduler-interval', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(f'{args.root_dir}/checkpoints', exist_ok=True)

    dataset = SupervisedModelDataset(args.data_dirs, normalize=True)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
    iterloader = get_batch(dataloader)

    in_features = dataset.cp_dim + dataset.ep_dim + dataset.op_dim
    out_features = dataset.op_dim
    model = Model(in_features, out_features)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.MSELoss()

    loss_stats = []
    model.train()
    for num_step in range(1, args.max_steps+1):
        loss = train_step(model, iterloader, criterion, optimizer)
        loss_stats.append(loss.item())

        if num_step % args.log_interval == 0:
            print(f'Step[{num_step}] {criterion}: {loss.item():.4f} lr: {get_lr(optimizer)}')
            plot_running_stats_step(repr(criterion), loss_stats, f'{args.root_dir}/loss-curve.png')

        if num_step % args.save_interval == 0:
            torch.save(model.state_dict(), f'{args.root_dir}/checkpoints/model_step={num_step}.pth')

        if num_step % args.scheduler_interval == 0:
            scheduler.step(min(loss_stats))
