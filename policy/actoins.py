from collections import defaultdict
import csv
from garage.experiment import Snapshotter
from garage import rollout
import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session
import numpy as np

experiment_path = 'data/local/experiment/seed1_itr500_bs5000/'
batch_num = 100

snapshotter = Snapshotter()
with tf.compat.v1.Session(): # optional, only for TensorFlow
    data = snapshotter.load(experiment_path)
    # Load the policy and the env in which it was trained
    policy = data['algo'].policy
    env = data['env']

    # See what the trained policy can accomplish
    batch_actions = []
    for i in range(batch_num):
        path = rollout(env, policy)  # path['env_infos']['step_action']: T*44
        batch_actions.append(path['env_infos']['step_action']) 
    
    batch_actions = np.array(batch_actions, dtype=object) # batch_num*T*44, T are different
    np.save('batch_actions.npy', batch_actions, allow_pickle=True)

# batch_actions = np.load('batch_actions.npy', allow_pickle=True)
# print(batch_actions[1].shape)


'''
# get log info, e.g. returns, we can also see logs from tensorboard
def csv2vec(path, prefix='./'):
    columns = defaultdict(list)  # each value in each column is appended to a list
    with open(prefix + path + 'progress.csv') as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            for (k, v) in row.items():  # go over each column name and value
                columns[k].append(v)  # append the value into the appropriate list based on column name k
    return columns

logs = csv2vec(experiment_path)
# print(logs.keys())
# print(logs['Evaluation/Iteration'])
# print(logs['Evaluation/MinReturn'])
# print(logs['TotalEnvSteps'])
print(logs['Evaluation/AverageReturn'])
# print(logs ['Evaluation/StdReturn'])
print(logs['Evaluation/MaxReturn'])
# print(logs['Extras/EpisodeRewardMean'])s
# print(logs['Evaluation/AverageDiscountedReturn'])  # discount factor is  1, AverageDiscountedReturn = AverageReturn
'''