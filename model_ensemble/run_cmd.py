import os
import random
import string


wd = 0
reparse_all_data = False


def id_generator(N=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))


def train_single_model(train_dirs, val_dirs):
    cmd = f'python train.py --max-iters 200000 --batch-size 256 --lr 0.001 --wd {wd} \
        --root-dir trained_models/single_model \
        --train-dirs {train_dirs} \
        --val-dirs {val_dirs}'
    if reparse_all_data:
        cmd += ' -FP'
    os.system(cmd)


def train_ensemble_model(train_dirs, val_dirs, N=10):
    model_id = id_generator()
    for n in range(N):
        cmd = f'python train.py --max-iters 10000 --batch-size 256 --lr 0.001 --wd {wd} \
            --root-dir trained_models/ensemble_model_{model_id}_child[{n}] \
            --train-dirs {train_dirs} \
            --val-dirs {val_dirs}'
        if reparse_all_data and n == 0:
            cmd += ' -FP'
        os.system(cmd)


if __name__ == '__main__':
    train_single_model()
    # train_ensemble_model()
