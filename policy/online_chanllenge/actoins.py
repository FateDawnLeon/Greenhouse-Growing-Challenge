# disable GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from garage.experiment import Snapshotter
from garage import rollout
import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session
import numpy as np
from env import GreenhouseSim
from garage.envs import GymEnv, normalize
from parameters import log_folder

prefix = 'data/local/model_final/'
experiment_path = prefix+log_folder

snapshotter = Snapshotter()
with tf.compat.v1.Session(): # optional, only for TensorFlow
    data = snapshotter.load(experiment_path)
    # # rollout
    # # See what the trained policy can accomplish
    # # Load the policy and the env in which it was trained
    policy = data['algo'].policy
    gh_env = GreenhouseSim(learning=False)
    env = normalize(GymEnv(gh_env))
    batch_actions = []
    batch_num = 100
    for i in range(batch_num):
        path = rollout(env, policy)  # path['env_infos']['step_action']: T*44
        # print(path.keys())
        # print(path['env_infos'].keys())
        len = path['env_infos']['agent_action'].shape[0]
        print(len)
        if len > 720 and len<1200:
            batch_actions.append(path['env_infos']) 
    
    # batch_actions = np.array(batch_actions, dtype=object) # batch_num*T*44, T are different
    # np.save('batch_paths.npy', batch_actions, allow_pickle=True)
    # print('saved')

    # batch_actions = np.load('batch_paths.npy', allow_pickle=True)
    action_num = 0
    for actions in batch_actions:
        json = GreenhouseSim.dump_actions(actions['agent_action'])
        with open(experiment_path+'/actions_'+str(action_num)+'.json', 'w') as f:
            f.write(json)
        action_num += 1