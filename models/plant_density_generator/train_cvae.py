import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from model import ConditionalVAE
from data import PlantDensityDataset, pd_str2feat_cvae


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--latent-size', type=int, default=3)
    parser.add_argument('--hidden-sizes', type=int, nargs="+", default=[32, 32])
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--root-dir', type=str, required=True)
    args = parser.parse_args()
    print(args)

    os.makedirs(args.root_dir, exist_ok=True)

    # prepare datasets
    dataset = PlantDensityDataset(args.num_samples, str2feat_func=pd_str2feat_cvae)
    val_size = round(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # prepare dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)

    # prepare model
    model = ConditionalVAE(
        input_size=dataset.feature_dim,
        latent_size=args.latent_size,
        hidden_sizes=args.hidden_sizes,
    )

    # prepare optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # KL divergence weights
    M_N_train = args.batch_size / len(train_dataset)
    M_N_val = args.batch_size / len(val_dataset)
    loss_best = 1e8

    for ep in range(1, args.max_epoch+1):
        model.train()
        for i, (inputs, num_days) in enumerate(train_loader):
            num_days = num_days.view(-1, 1)
            outputs = model(inputs, labels=num_days)
            result = model.loss_function(*outputs, M_N=M_N_train)
            
            loss = result['loss']
            recon_loss = result['Reconstruction_Loss']
            kld = result['KLD']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch[{ep}/{args.max_epoch}] Batch[{i+1}/{len(train_loader)}] loss: {loss.item():.8f}, recon_loss: {recon_loss:.8f}, KLD: {kld:.8f}")

        model.eval()
        loss_val = 0
        for j, (inputs, num_days) in enumerate(val_loader):
            with torch.no_grad():
                num_days = num_days.view(-1, 1)
                outputs = model(inputs, labels=num_days)
                loss = model.loss_function(*outputs, M_N=M_N_val)['loss']
            loss_val += loss.item() * inputs.size(0)
        loss_val /= len(val_dataset)
        
        print(f"Epoch[{ep}/{args.max_epoch}] val_loss: {loss.item():.8f}")

        torch.save(model.state_dict(), f"{args.root_dir}/model-ep={ep}.pth")

        if loss_val < loss_best:
            loss_best = loss_val
            checkpoint = {'state_dict': model.state_dict(), 'ep': ep, 'loss': loss_best}
            torch.save(checkpoint, f"{args.root_dir}/checkpoint_best.pth")
