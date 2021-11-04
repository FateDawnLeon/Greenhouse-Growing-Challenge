from collections import defaultdict
import csv
import matplotlib.pyplot as plt
from parameters import experiment_path

# get log info, e.g. returns, we can also see logs from tensorboard
def progress(path=experiment_path, prefix='./'):
    logs = defaultdict(list)  # each value in each column is appended to a list
    with open(prefix + path + '/progress.csv') as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            for (k, v) in row.items():  # go over each column name and value
                logs[k].append(v)  # append the value into the appropriate list based on column name k

    # logs Evaluation key:
    # 'Evaluation/Iteration', 'Evaluation/NumEpisodes', 'Evaluation/StdReturn', 'Evaluation/TerminationRate',
    # 'Evaluation/AverageReturn', 'Evaluation/AverageDiscountedReturn', # discount factor is  1, AverageDiscountedReturn = AverageReturn
    # 'Evaluation/MaxReturn', 'Evaluation/MinReturn', 

    f=plt.figure()
    plt.plot([float(i) for i in logs['Evaluation/Iteration']], [float(i) for i in logs['Evaluation/AverageReturn']], label='Average')
    print('Max AverageReturn:', max([float(i) for i in logs['Evaluation/AverageReturn']]))
    plt.plot([float(i) for i in logs['Evaluation/Iteration']], [float(i) for i in logs['Evaluation/MaxReturn']], label='Max')
    print('Max MaxReturn:', max([float(i) for i in logs['Evaluation/MaxReturn']]))
    plt.plot([float(i) for i in logs['Evaluation/Iteration']], [float(i) for i in logs['Evaluation/MinReturn']], label='Min')
    plt.plot([float(i) for i in logs['Evaluation/Iteration']], [float(i) for i in logs['Evaluation/StdReturn']], label='Std')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Return')
    f.savefig(prefix + path + '/progress.png', bbox_inches='tight')
    # plt.show()

progress()