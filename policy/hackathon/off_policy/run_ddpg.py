#!/usr/bin/env python3
"""This is a script to train a greenhouse sim with DDPG algorithm.

To use it add follwing to .bashrc
export PYTHONPATH="${PYTHONPATH}:/your-path-to-Greenhouse-Growing-Challenge/models/model_hackathon"
e.g. export PYTHONPATH="${PYTHONPATH}:/home/liuys/Greenhouse-Growing-Challenge/models/model_hackathon"
"""
# disable GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from env import GreenhouseSim

import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler, RaySampler, DefaultWorker, VecWorker, FragmentWorker
from ddpg import DDPG
from garage.tf.policies import GaussianMLPPolicy, GaussianLSTMPolicy, ContinuousMLPPolicy
from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise, AddGaussianNoise, epsilon_greedy_policy
from path_buffer import PathBuffer
from her_replay_buffer import HERReplayBuffer
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.trainer import TFTrainer
from parameters import hyper as h
from parameters import prefix

import psutil

@wrap_experiment(prefix=prefix, name=f'{h}', snapshot_mode='all')  # snapshot_mode='last'/'all'
def rl_greenhouse(ctxt, pl=h['pl'], pls=h['pls'], qfs=h['qfs'], buffer=h['buffer'], expl=h['expl'],\
     n_cycles=h['n_cycles'], n_epochs=h['n_epochs'], batch_size=h['batch_size'], seed=h['seed'], grad_gain=h['grad_gain']):
    """Train DDPG with greenhouse sim.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        gh_env = GreenhouseSim(training=True, grad_gain=grad_gain)
        env = normalize(GymEnv(gh_env))

        if pl == 'cont':
             policy = ContinuousMLPPolicy(env_spec = env.spec,
                                     hidden_sizes = pls,
            )
        elif pl == 'mlp':
            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes = pls
            )

        if expl == 'Ornstein':
            exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec, policy, sigma=0.2)
        elif expl == 'Gaussian':
            exploration_policy = AddGaussianNoise(env.spec, policy)
        elif expl == 'epsilon':
            exploration_policy = epsilon_greedy_policy(env.spec, policy)
        
        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes = qfs)


        if buffer == 'path':
            replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
        elif buffer == 'her':
            replay_buffer = HERReplayBuffer(capacity_in_transitions=int(1e6))

        sampler = RaySampler(agents=exploration_policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True,
                             worker_class=FragmentWorker,
                             worker_args=dict(n_envs=32),
                             n_workers=psutil.cpu_count(logical=True)
                             )

        #  LocalSample debug
        # sampler = LocalSampler(agents=exploration_policy,
        #                        envs=env,
        #                        max_episode_length=env.spec.max_episode_length,
        #                        is_tf_worker=True,
        #                         # worker_class=FragmentWorker
        #                        )

        ddpg = DDPG(env_spec=env.spec,
                    policy=policy,
                    policy_lr=1e-4,
                    qf_lr=1e-3,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    sampler=sampler,
                    steps_per_epoch=n_cycles,
                    target_update_tau=1e-2,
                    n_train_steps=50,
                    discount=1,
                    min_buffer_size=int(1e4),
                    exploration_policy=exploration_policy,
                    policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                    qf_optimizer=tf.compat.v1.train.AdamOptimizer)

        trainer.setup(algo=ddpg, env=env)
        trainer.train(n_epochs=n_epochs, batch_size=batch_size, store_episodes=True)

rl_greenhouse()