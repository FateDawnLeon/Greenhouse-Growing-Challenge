hyper = {
    'alg': 'DDPG',  # DDPG
    'pl': 'cont',  # cont, mlp
    'pls': [128, 32],  # policy hidden size
    'expl': 'Ornstein',  # Gaussian, epsilon
    'qfs': [128, 32],  # q function hidden size
    'buffer': 'path',  # her
    'n_cycles': 50,
    'n_epochs': 500,
    'batch_size': 5000,  
    'seed': 1,
    'grad_gain': False
}

data_path = 'data/local/'
prefix = 'model_batch7'
experiment_path = f'{data_path}{prefix}/{hyper}'