import os
import torch
import argparse

from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, optimizer, lr_scheduler
from utils import save_json_data, plot_loss_curve
from data import ClimateDatasetHour, PlantDatasetHour, ClimateDatasetDay, PlantDatasetDay
from model import ClimateModel, PlantModel, ClimateModelDay, PlantModelDay
from constant import PLANT_NORM_DATA


OPTIM = {
    'adam': Adam,
    'sgd': SGD,
}

DATASET = {
    'climate_hour': ClimateDatasetHour,
    'plant_hour': PlantDatasetHour,
    'climate_day': ClimateDatasetDay,
    'plant_day': PlantDatasetDay,
}

MODEL = {
    'climate_hour': ClimateModel,
    'plant_hour': PlantModel,
    'climate_day': ClimateModelDay,
    'plant_day': PlantModelDay,
}

NORM_DATA = {
    "plant_day": PLANT_NORM_DATA
}

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


def train_step(model, iterloader, optimizer, criterion):
    input, target = next(iterloader)
    pred = model(*input)
    loss = criterion(pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def validate(model, val_loader, criterion):
    running_loss = 0
    for input, target in val_loader:
        with torch.no_grad():
            pred = model(*input)
            loss = criterion(pred, target)
        running_loss += loss.item() * target.size(0)
    return running_loss / len(val_loader.dataset)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, required=True)
    parser.add_argument('--train-dirs', nargs='+', required=True)
    parser.add_argument('--val-dirs', nargs='+', required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-patience', type=int, default=5)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--max-iters', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--val-interval', type=int, default=1000)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('-PLS', '--plot-log-scale', action='store_true')
    parser.add_argument('-FP', '--force-preprocess', action='store_true')
    parser.add_argument('-FT', '--finetune', action='store_true')
    parser.add_argument('-CP', '--ckpt-path', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)

    os.makedirs(f'{args.root_dir}/checkpoints', exist_ok=True)
    save_json_data(vars(args), f"{args.root_dir}/config.json")

    MyDataset = DATASET[args.model]
    # norm_data = MyDataset.get_norm_data(args.train_dirs + args.val_dirs)
    norm_data = NORM_DATA[args.model]

    train_dataset = MyDataset(args.train_dirs, norm_data, force_preprocess=args.force_preprocess)
    val_dataset = MyDataset(args.val_dirs, norm_data, force_preprocess=args.force_preprocess)

    iter_loader = get_batch(get_dataloader(train_dataset))
    val_loader = get_dataloader(val_dataset, is_train=False)

    model = MODEL[args.model](**train_dataset.meta_data)

    if args.finetune:
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['state_dict'])

    optimizer = OPTIM[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=args.min_lr, patience=args.lr_patience)
    criterion = torch.nn.MSELoss()

    loss_stats = {'train':[], 'val':[]}
    model.train()
    for step in range(1, args.max_iters+1):
        loss = train_step(model, iter_loader, optimizer, criterion)
        loss_stats['train'].append((step, loss))

        if step % args.log_interval == 0:
            print(f'Iter[{step}/{args.max_iters}] Train Loss: {loss:.6f} | lr: {get_lr(optimizer)}')

        if step % args.val_interval == 0:
            model.eval()
            loss_val = validate(model, val_loader, criterion)
            model.train()
            
            scheduler.step(loss_val)
            
            print(f'Iter[{step}/{args.max_iters}] Val Loss: {loss_val:.6f}')
            
            torch.save({'state_dict': model.state_dict(), 'norm_data': norm_data}, 
                f'{args.root_dir}/checkpoints/step={step}.pth')

            loss_stats['val'].append((step, loss_val))
            save_json_data(loss_stats, f'{args.root_dir}/loss_stats.json')
            plot_loss_curve(loss_stats, f'{args.root_dir}/loss_curve.png', args.plot_log_scale)
