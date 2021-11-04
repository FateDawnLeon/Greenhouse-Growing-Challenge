# disable GPU
import glob, os
import json, requests
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from garage.experiment import Snapshotter
from garage import rollout
import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session
import numpy as np
from env import GreenhouseSim
from garage.envs import GymEnv, normalize
from parameters import experiment_path
from data import dump_actions
from utils import load_json_data

eval_policy_dir = f'{experiment_path}/eval_policy'

def predict_actions(num_trials, policy_itr = 30):
    batch_actions = []
    snapshotter = Snapshotter()
    with tf.compat.v1.Session(): # optional, only for TensorFlow
        data = snapshotter.load(experiment_path, itr=policy_itr)
        # # rollout
        # # See what the trained policy can accomplish
        # # Load the policy and the env in which it was trained
        policy = data['algo'].policy
        gh_env = GreenhouseSim(training=False)
        env = normalize(GymEnv(gh_env))
        
        for i in range(num_trials):
            path = rollout(env, policy)  # path['env_infos']['parsed_action']: day*17
            # print(path.keys())
            # print(path['env_infos'].keys())
            len = path['env_infos']['parsed_action'].shape[0]
            print(len)
            if len > 25:
                batch_actions.append(path['env_infos']['parsed_action']) 
    
    batch_actions = np.array(batch_actions) # batch_num*day*17, day are different

    # np.save('batch_paths.npy', batch_actions, allow_pickle=True)
    # print('saved')
    # batch_actions = np.load('batch_paths.npy', allow_pickle=True)
 
    for i in range(batch_actions.shape[0]):
        os.makedirs(eval_policy_dir, exist_ok=True)
        dump_actions(f'{eval_policy_dir}/actions_{i}.json', batch_actions[i])

KEYS = {
    'hack': 'H17-KyEO-iDtD-mVGR'
}
URL = 'https://www.digigreenhouse.wur.nl/Kasprobeta/'
def query_simulator(control, sim_id):
    data = {"key": KEYS[sim_id], "parameters": json.dumps(control)}
    headers = {'ContentType': 'application/json'}

    while True:
        response = requests.post(URL, data=data, headers=headers, timeout=300)
        output = response.json()
        print(response, output['responsemsg'])

        if output['responsemsg'] == 'ok':
            break
        elif output['responsemsg'] == 'busy':
            continue
        else:
            raise ValueError('response message not expected!')
    
    return output

def eval_actions():
    control_list = glob.glob(f'{eval_policy_dir}/actions_*.json')
    print(control_list)
    for f in control_list:
        control = load_json_data(f)
        output = query_simulator(control, sim_id='hack')

        balance = output['stats']['economics']['balance']
        print('best netprofit of final submission:', balance)

predict_actions(10, policy_itr = 30)
eval_actions()
