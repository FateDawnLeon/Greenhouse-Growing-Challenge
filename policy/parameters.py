hyper = {
    'alg': 'TRPO',  # PPO
    'pl': 'mlp',  # lstm
    'pls': (32, 32),  # policy hidden size
    'bl': 'Gaussian',  # baseline: Linear
    'bls': (32, 32),  # baseline hidden size
    'n_epochs': 500,
    'batch_size': 5000,  
    'seed': 1
}
hyper['clip'] = 0.01 if hyper['alg'] == 'TRPO' else 0.2  # TRPO default = 0.01; PPO default = 0.2

log_folder = hyper['alg']+str(hyper['clip'])+'_'+hyper['pl']+str(hyper['pls'][0])+'_'+str(hyper['pls'][1])\
            +'_'+hyper['bl']+str(hyper['bls'][0])+'_'+str(hyper['bls'][1])\
            +'_itr'+str(hyper['n_epochs'])+'_bs'+str(hyper['batch_size'])+'_sd'+str(hyper['seed'])