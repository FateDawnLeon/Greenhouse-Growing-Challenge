hyper = {
    'alg': 'DDPG',  # DDPG
    'pl': 'cont',  # cont, mlp
    'pls': [256, 128, 64],  # policy hidden size
    'expl': 'Ornstein',  # Gaussian, epsilon
    'qfs': [256, 128, 64],  # q function hidden size
    'buffer': 'path',  # her
    'n_cycles': 50,
    'n_epochs': 500,
    'batch_size': 5000,  
    'seed': 1
}

data_path = 'data/local/'
prefix = 'model_batch5'
experiment_path = f'{data_path}{prefix}/{hyper}'