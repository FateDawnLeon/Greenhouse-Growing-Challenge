hyper = {
    'alg': 'TRPO',  # PPO, TRPO
    'pl': 'mlp',  # mlp, lstm
    'pls': [256, 128, 64],  # policy hidden size
    'bl': 'Gaussian',  # baseline: Linear
    'bls': [256, 128, 64],  # baseline hidden size
    'batch_size': 5000,
    'n_epochs': 500,
    'seed': 1, 
}
hyper['clip'] = 0.01 if hyper['alg'] == 'TRPO' else 0.2  # TRPO default = 0.01; PPO default = 0.2