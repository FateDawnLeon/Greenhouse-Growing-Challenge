#!/usr/bin/env python3
"""This is a script to train a greenhouse sim with DDPG algorithm.

To use it add follwing to .bashrc
export PYTHONPATH="${PYTHONPATH}:/your-path-to-Greenhouse-Growing-Challenge/model_forward"
e.g. export PYTHONPATH="${PYTHONPATH}:/home/liuys/Greenhouse-Growing-Challenge/model_forward"
"""
# disable GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from env import GreenhouseSim

import click
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler, RaySampler, DefaultWorker, VecWorker, FragmentWorker
from garage.tf.algos import TRPO, PPO
from ddpg import DDPG
from garage.tf.policies import GaussianMLPPolicy, GaussianLSTMPolicy, ContinuousMLPPolicy
from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise, AddGaussianNoise, epsilon_greedy_policy
from bath_buffer import PathBuffer
from her_replay_buffer import HERReplayBuffer
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.trainer import TFTrainer
from parameters import hyper, log_folder

@click.command()
@click.option('--pl', default=hyper['pl'])
@click.option('--pls0', default=hyper['pls'][0])
@click.option('--pls1', default=hyper['pls'][1])
@click.option('--qfs0', default=hyper['qfs'][0])
@click.option('--qfs1', default=hyper['qfs'][1])
@click.option('--n_epochs', default=hyper['n_epochs'])
@click.option('--n_cycles', default=hyper['n_cycles'])
@click.option('--buffer', default=hyper['buffer'])
@click.option('--expl', default=hyper['expl'])
@click.option('--batch_size', default=hyper['batch_size'])
@click.option('--seed', default=hyper['seed'])
@wrap_experiment(prefix='model_final', name=log_folder, snapshot_mode='last')  # snapshot_mode='last'/'all'
def rl_greenhouse(ctxt, pl, pls0, pls1, qfs0, qfs1, buffer, expl, n_cycles, n_epochs, batch_size, seed):
    """Train DDPG with greenhouse sim.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        gh_env = GreenhouseSim()
        env = normalize(GymEnv(gh_env))

        if pl == 'cont':
             policy = ContinuousMLPPolicy(env_spec=env.spec,
                                     hidden_sizes=(pls0, pls1),
            )
        elif pl == 'mlp':
            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(pls0, pls1),
            )
        elif pl == 'lstm':
            policy = GaussianLSTMPolicy(
                env_spec=env.spec,
                hidden_dim=pls0,
            )

        if expl == 'Ornstein':
            exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec, policy, sigma=0.2)
        elif expl == 'Gaussian':
            exploration_policy = AddGaussianNoise(env.spec, policy)
        elif expl == 'epsilon':
            exploration_policy = epsilon_greedy_policy(env.spec, policy)
        
        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=(qfs0, qfs1))


        if buffer == 'path':
            replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
        elif buffer == 'her':
            replay_buffer = HERReplayBuffer(capacity_in_transitions=int(1e6))

        sampler = RaySampler(agents=exploration_policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True,
                             worker_class=FragmentWorker,
                             worker_args=dict(n_envs=6),
                            #  n_workers=96
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