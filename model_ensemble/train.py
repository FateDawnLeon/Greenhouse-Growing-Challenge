import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, optimizer, lr_scheduler

from model import AGCModel
from data import AGCDataset, get_norm_data


OPTIM = {
    'adam': Adam,
    'sgd': SGD,
}


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


def get_batch(dataloader):
    while True:
        yield from dataloader


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_dataloader(dataset, is_train=True):
    return DataLoader(dataset,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=is_train,
            pin_memory=True,
            drop_last=is_train)


def train_step(model, iterloader, optimizer):
    input, target = next(iterloader)
    loss = model.loss(input, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def validate(model, val_loader):
    model.eval()
    running_loss = 0
    for input, target in val_loader:
        with torch.no_grad():
            loss = model.loss(input, target)
        running_loss += loss.item() * target.size(0)
    return running_loss / len(val_loader.dataset)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--max-iters', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--train-dirs', nargs='+', required=True)
    parser.add_argument('--val-dirs', nargs='+', required=True)
    parser.add_argument('--root-dir', type=str, required=True)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--val-interval', type=int, default=1000)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('-FP', '--force-preprocess', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.makedirs(f'{args.root_dir}/checkpoints', exist_ok=True)

    train_dataset = AGCDataset(args.train_dirs, force_preprocess=args.force_preprocess)
    val_dataset = AGCDataset(args.val_dirs, force_preprocess=args.force_preprocess)
    norm_data = get_norm_data(train_dataset)
    iter_loader = get_batch(get_dataloader(train_dataset))
    val_loader = get_dataloader(val_dataset, is_train=False)
    model = AGCModel(
        cp_dim=train_dataset.cp_dim, 
        ep_dim=train_dataset.ep_dim, 
        op_dim=train_dataset.op_dim,
        norm_data=norm_data,
    )

    optimizer = OPTIM[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5)

    for step in range(1, args.max_iters+1):
        loss = train_step(model, iter_loader, optimizer)

        if step % args.log_interval == 0:
            print(f'Iter[{step}/{args.max_iters}] Train Loss: {loss:.4f} | lr: {get_lr(optimizer)}')

        if step % args.val_interval == 0:
            loss_val = validate(model, val_loader)
            print(f'Iter[{step}/{args.max_iters}] Val Loss: {loss_val:.4f}')
            scheduler.step(loss_val)
            torch.save(
                {'state_dict': model.state_dict(), 'norm_data': norm_data}, 
                f'{args.root_dir}/checkpoints/step={step}-train_loss={loss:.4f}-loss_val={loss_val:.4f}.pth'
            )