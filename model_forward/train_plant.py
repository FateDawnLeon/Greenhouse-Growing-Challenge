import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, optimizer, lr_scheduler

from model import ModelPlant
from data import AGCDatasetPlant, preprocess_data_plant, compute_mean_std_plant

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
plt.ioff()


class Logger(object):
    def __init__(self, print_interval, num_batch):
        super().__init__()
        self.step = 0
        self.loss_stats = []
        self.print_interval = print_interval
        self.num_bacth = num_batch

    def log(self, loss, lr):
        self.loss_stats.append(loss)
        self.step += 1

        if self.step % self.print_interval == 0:
            batch_idx = self.step % self.num_bacth
            epoch_idx = self.step // self.num_bacth + 1
            print(f'Epoch[{epoch_idx}] Batch[{batch_idx}/{self.num_bacth}] Loss[{loss:.4f}] lr[{lr}]')


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


def get_dataloaders(args, norm_data):
    train_dataset = AGCDatasetPlant(args.train_dirs, norm_data=norm_data)
    val_dataset = AGCDatasetPlant(args.val_dirs, norm_data=norm_data)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=8,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size,
                            num_workers=8,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)

    return train_loader, val_loader


def get_optimizer(model, args):
    if args.optimizer == 'adam':
        return Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        return SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError()


def train_epoch(model, train_loader, criterion, optimizer, logger):
    model.train()
    running_loss = []
    for cp, ep, op_other, op_plant_pre, op_plant_cur in train_loader:
        prediction = model(cp, ep, op_other, op_plant_pre)
        loss = criterion(prediction, op_plant_cur)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log(loss.item(), get_lr(optimizer))
        running_loss.append(loss.item())
    
    return sum(running_loss) / len(running_loss)


def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0
    for cp, ep, op_other, op_plant_pre, op_plant_cur in val_loader:
        with torch.no_grad():
            prediction = model(cp, ep, op_other, op_plant_pre)
            loss = criterion(prediction, op_plant_cur)
        running_loss += loss.item() * cp.size(0)
    return running_loss / len(val_loader.dataset)


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--lr-milestones', type=int, nargs='+', default=[50, 75])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--train-dirs', nargs='+', required=True)
    parser.add_argument('--val-dirs', nargs='+', required=True)
    parser.add_argument('--root-dir', type=str, required=True)
    parser.add_argument('--print-interval', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('-FP', '--force-preprocess', action='store_true')
    args = parser.parse_args()

    os.makedirs(f'{args.root_dir}/checkpoints', exist_ok=True)

    for data_dir in args.train_dirs + args.val_dirs:
        if not os.path.exists(f'{data_dir}/processed_data_plant.npz') or args.force_preprocess:
            preprocess_data_plant(data_dir)

    norm_data = compute_mean_std_plant(args.train_dirs + args.val_dirs)
    train_loader, val_loader = get_dataloaders(args, norm_data)
    model = ModelPlant(
        op_other_dim=train_loader.dataset.op_other_dim, 
        op_plant_dim=train_loader.dataset.op_plant_dim,
        norm_data=norm_data
    )

    optimizer = get_optimizer(model, args)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    criterion = nn.MSELoss()

    logger = Logger(args.print_interval, len(train_loader))

    for epoch in range(1, args.max_epochs+1):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, logger)
        loss_val = validate(model, val_loader, criterion)
        scheduler.step()

        print(f'Epoch[{epoch}] Train Loss: {loss_train:.4f} | Val Loss: {loss_val:.4f}')
        
        torch.save(
            {'state_dict': model.state_dict(), 'norm_data': norm_data}, 
            f'{args.root_dir}/checkpoints/epoch={epoch}-loss_train={loss_train:.4f}-loss_val={loss_val:.4f}.pth'
        )
