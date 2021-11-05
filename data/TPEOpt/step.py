from space import SPACES
from search import objective

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default=None)
    parser.add_argument('-controls', type=str, default=None)
    parser.add_argument('-to_day', type=int, default=1)
    args = parser.parse_args()

    objective(SPACES[args.controls], False, args.mode, args.to_day, bo=False)
