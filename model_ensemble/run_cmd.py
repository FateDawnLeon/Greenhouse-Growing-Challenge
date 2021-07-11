import os
import random
import string

from data import preprocess_data
from constant import ENV_KEYS, OUTPUT_KEYS


wd = 1e-4


def id_generator(N=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))


def preprocess(data_dirs):
    for data_dir in data_dirs.split():
        preprocess_data(data_dir, 'processed_data', ENV_KEYS, OUTPUT_KEYS)


def train_single_model(train_dirs, val_dirs, reparse_data=False):
    cmd = f'python train.py --max-iters 200000 --batch-size 256 --lr 0.001 --wd {wd} \
        --root-dir trained_models/single_model \
        --train-dirs {train_dirs} \
        --val-dirs {val_dirs}'
    if reparse_data:
        cmd += ' -FP'
    os.system(cmd)


def train_ensemble_model(train_dirs, val_dirs, N, reparse_data=False):
    if reparse_data:
        preprocess(train_dirs)
        preprocess(val_dirs)
    
    model_id = id_generator()
    for n in range(N):
        model_name = f'ensemble_model_{model_id}_child[{n}]'
        cmd = f'python train.py --max-iters 50000 --batch-size 1024 --lr 0.001 --wd {wd} \
            --root-dir trained_models/{model_name} \
            --train-dirs {train_dirs} \
            --val-dirs {val_dirs} \
            --gpu {n}'
        cmd = f'nohup {cmd} > trained_models/{model_name}.log 2>&1 &'
        os.system(cmd)
        print(f'{model_name} starts training...')


if __name__ == '__main__':
    reparse_data = True
    prefix = '/home/liuys/Greenhouse-Growing-Challenge/collect_data/data_sample/sim=A/'

    train_folders = [
        'data_sample=BO_data_2021-07-07_SIM=A_DS=DPD_OPT=gbrt_NI=500_NC=5000_P=0',
        'data_sample=BO_data_2021-07-08_SIM=A_DS=BSTPD_OPT=gbrt_NI=500_NC=5000_P=0',
        'data_sample=BO_date=0626_sim=A_method=gp_init=200_ncall=1000',
        'data_sample=BO_date=0626_sim=A_space=B_N=1000_init=100',
        'data_sample=BO_date=0627_sim=A_method=gbrt_init=100_ncall=1000_RS=1234',
        'data_sample=BO_date=0628_method=gbrt_init=1000_ncall=10000_RS=42_SP=A_P=1_sim=A',
        'data_sample=BO_date=0628_method=gbrt_init=100_ncall=1000_RS=1_SP=F_P=1_sim=A',
        'data_sample=BO_date=0628_sim=A_method=gbrt_init=100_ncall=1000_RS=12345_SP=E_P=1',
        'data_sample=BO_date=2021-06-30_SIM=A_DS=AA_OPT=gbrt_NI=500_NC=5000_P=0',
        'data_sample=BO_date=2021-07-08_SIM=A_DS=A4BSTPD2_OPT=gbrt_NI=100_NC=1000_P=0',
        'data_sample=BO_date=2021-07-08_SIM=A_DS=B3_OPT=gbrt_NI=10_NC=100_P=2',
        'data_sample=BO_date=2021-07-08_SIM=A_DS=B3_OPT=gbrt_NI=30_NC=300_P=2',
        'data_sample=BO_date=2021-07-09_SIM=A_DS=A4PD2_OPT=gbrt_NI=100_NC=1000_P=0',
        'data_sample=BO_date=2021-07-09_SIM=A_DS=A4PD2_OPT=gbrt_NI=500_NC=5000_P=0',
        'data_sample=grid_search_date=2021-06-19_sim=A_number=360',
        'data_sample=original_random_date=2021-06-19_sim=A_number=1000',
        'data_sample=original_random_date=2021-06-20_sim=A_number=2000',
        'data_sample=original_random_date=2021-06-21_sim=A_number=5000',
        'data_sample=original_random_date=2021-06-23_sim=A_number=5000',
        'data_sample=original_random_date=2021-06-24_sim=A_number=5000',
        'data_sample=original_random_date=2021-06-25_sim=A_number=8869',
        'data_sample=random_date=2021-07-05_sim=A_number=100',
        'data_sample=random_date=2021-07-06_sim=A_number=1000'
    ]

    train_dirs = ' '.join(f'{prefix}{f}' for f in train_folders)

    val_folders = ['data_sample=original_random_date=2021-06-22_sim=A_number=10000']
    val_dirs = ' '.join(f'{prefix}{f}' for f in val_folders)

    # train_single_model(train_dirs, val_dirs, reparse_data=reparse_data)
    train_ensemble_model(train_dirs, val_dirs, N=8, reparse_data=reparse_data)
