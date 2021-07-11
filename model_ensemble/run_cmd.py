import os
import random
import string


wd = 0
reparse_all_data = False


def id_generator(N=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))


def train_single_model():
    cmd = f'python train.py --max-iters 200000 --batch-size 256 --lr 0.001 --wd {wd} \
        --root-dir trained_models/single_model \
        --train-dirs YOUR_OWN_DIRS \
        --val-dirs YOUR_OWN_DIRS'
    os.system(cmd)


def train_ensemble_model(N=10):
    model_id = id_generator()
    for n in range(N):
        cmd = f'python train.py --max-iters 10000 --batch-size 256 --lr 0.001 --wd {wd} \
            --root-dir trained_models/ensemble_model_{model_id}_child[{n}] \
            --train-dirs YOUR_OWN_DIRS \
            --val-dirs YOUR_OWN_DIRS'
        os.system(cmd)


if __name__ == '__main__':
    train_single_model()
    # train_ensemble_model()
