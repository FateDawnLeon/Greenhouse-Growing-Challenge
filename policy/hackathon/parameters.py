hyper = {
    'sim': 'hack',  # hackathon
    'alg': 'DDPG',  # PPO, TRPO, DDPG
    'pl': 'cont',  # cont, mlp, lstm
    'pls': (64, 64),  # policy hidden size
    'n_epochs': 500,
    'batch_size': 5000,  
    'seed': 1,
    # DDPG
    'n_cycles': 50,
    'qfs': (64, 64),  # q function hidden size
    'buffer': 'path',  # her
    'expl': 'Ornstein',  # Gaussian, epsilon
    # TRPO/PO
    'bl': 'Gaussian',  # baseline: Linear
    'bls': (32, 32),  # baseline hidden size
}
hyper['clip'] = 0.01 if hyper['alg'] == 'TRPO' else 0.2  # TRPO default = 0.01; PPO default = 0.2

if hyper['alg'] in ['TRPO', 'PPO']:
    log_folder = hyper['sim']+'_'+hyper['alg']+str(hyper['clip'])+'_'+hyper['pl']+str(hyper['pls'][0])+'_'\
            +str(hyper['pls'][1]) +'_'+hyper['bl']+str(hyper['bls'][0])+'_'+str(hyper['bls'][1])\
            +'_itr'+str(hyper['n_epochs'])+'_bs'+str(hyper['batch_size'])+'_sd'+str(hyper['seed'])
elif hyper['alg'] == 'DDPG':
    log_folder =  hyper['sim']+'_'+hyper['alg']+'_'+hyper['buffer']+'_'+hyper['expl']+'_'+hyper['pl']+str(hyper['pls'][0])+'_'+str(hyper['pls'][1]) +'_qf'\
                +str(hyper['bls'][0])+'_'+str(hyper['bls'][1]) +'_itr'+str(hyper['n_epochs'])\
                +'_bs'+str(hyper['batch_size'])+'_sd'+str(hyper['seed'])