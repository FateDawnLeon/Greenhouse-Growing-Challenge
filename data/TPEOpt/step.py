from space import SPACES
from search import objective

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-controls', type=str, default=None)
    parser.add_argument('-auto', action='store_true')
    parser.add_argument('-mode', type=str, default=None)
    parser.add_argument('-to_day', type=int, default=1)
    args = parser.parse_args()

    num_day = SPACES[args.controls]['duration']
    if args.auto:
        objective(SPACES[args.controls], 'pause', num_day, bo=False)
        # objective(SPACES[args.controls], 'step', int(num_day/2), bo=False)
        # objective(SPACES[args.controls], 'step', num_day, bo=False)
        # n_piece = 10
        # for i in range(n_piece):
        #     objective(SPACES[args.controls], 'step', int(num_day/n_piece*(i+1)), bo=False)
    else:
        objective(SPACES[args.controls], args.mode, args.to_day, bo=False)
