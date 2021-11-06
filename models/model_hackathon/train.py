import os
import torch
import argparse

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam, lr_scheduler
from utils import save_json_data, plot_loss_curve
from data import ClimateDatasetDay, PlantDatasetDay
from model import MODEL_CLASSES, MODEL_CONFIGS


OPTIM = {
    'adam': Adam,
    'sgd': SGD,
}

DATASET = {
    'climate_day': ClimateDatasetDay,
    'plant_day': PlantDatasetDay,
}


def get_batch(dataloader):
    while True:
        yield from dataloader


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_dataloader(batch_size, dataset, is_train=True):
    return DataLoader(dataset,
            batch_size=batch_size,
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


def split_dataset(dataset, val_ratio):
    num_all = len(dataset)
    num_train = round(num_all * (1 - val_ratio))
    num_val = num_all - num_train
    return random_split(dataset, [num_train, num_val], generator=torch.Generator().manual_seed(42))


def save_checkpoint(model, model_class, model_config, save_path):
    ckpt = {
        'state_dict': model.state_dict(),
        'model_class': model_class,
        'model_config': model_config,
    }
    torch.save(ckpt, save_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--root-dir', type=str, required=True)
    parser.add_argument('-DD', '--data-dirs', nargs='+', required=True)
    parser.add_argument('-M', '--model', type=str, required=True)
    parser.add_argument('-MC', '--model-config', type=str, default='A0')
    parser.add_argument('-VR', '--val-ratio', type=float, default=0.2)
    parser.add_argument('-CF', '--control-folder', type=str, default="controls")
    parser.add_argument('-OF', '--output-folder', type=str, default="outputs")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('-LP', '--lr-patience', type=int, default=5)
    parser.add_argument('-ML', '--min-lr', type=float, default=1e-5)
    parser.add_argument('-MI', '--max-iters', type=int, default=20000)
    parser.add_argument('-BS', '--batch-size', type=int, default=100)
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

    dataset = DATASET[args.model](
        data_dirs=args.data_dirs, 
        force_preprocess=args.force_preprocess,
        control_folder=args.control_folder,
        output_folder=args.output_folder)

    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio)
    iter_loader = get_batch(get_dataloader(args.batch_size, train_dataset))
    val_loader = get_dataloader(args.batch_size, val_dataset, is_train=False)

    model_config = MODEL_CONFIGS[args.model_config]
    model_class = MODEL_CLASSES[args.model]
    model_config.update(dataset.get_meta_data())
    model = model_class(**model_config)

    print(model)

    if args.finetune:
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['state_dict'])

    optimizer = OPTIM[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=args.min_lr, patience=args.lr_patience)
    criterion = torch.nn.MSELoss()

    loss_stats = {'train':[], 'val':[]}
    loss_best = 1e8
    
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
            
            save_path = f"{args.root_dir}/checkpoints/step-{step}.pth"
            save_checkpoint(model, args.model, model_config, save_path)
            
            if loss_val < loss_best:
                loss_best = loss_val
                save_path = f"{args.root_dir}/ckpt_best.pth"
                save_checkpoint(model, args.model, model_config, save_path)

            loss_stats['val'].append((step, loss_val))
            save_json_data(loss_stats, f'{args.root_dir}/loss_stats.json')
            plot_loss_curve(loss_stats, f'{args.root_dir}/loss_curve.png', args.plot_log_scale)
