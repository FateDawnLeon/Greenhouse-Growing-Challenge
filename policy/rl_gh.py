#!/usr/bin/env python3
"""This is a script to train a greenhouse sim with RL algorithm.

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
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.baselines import GaussianMLPBaseline
from garage.sampler import LocalSampler, RaySampler, DefaultWorker, VecWorker
from garage.tf.algos import TRPO, PPO
from garage.tf.policies import GaussianMLPPolicy, GaussianLSTMPolicy
from garage.trainer import TFTrainer
from parameters import hyper, log_folder

@click.command()
@click.option('--alg', default=hyper['alg'])
@click.option('--clip', default=hyper['clip'])  # TRPO default = 0.01; PPO default = 0.2
@click.option('--pl', default=hyper['pl'])
@click.option('--pls0', default=hyper['pls'][0])
@click.option('--pls1', default=hyper['pls'][1])
@click.option('--bl', default=hyper['bl'])
@click.option('--bls0', default=hyper['bls'][0])
@click.option('--bls1', default=hyper['bls'][1])
@click.option('--n_epochs', default=hyper['n_epochs'])
@click.option('--batch_size', default=hyper['batch_size'])
@click.option('--seed', default=hyper['seed'])
@wrap_experiment(name=log_folder, snapshot_mode='all')  # snapshot_mode='last'
def rl_greenhouse(ctxt, alg, clip, pl,  pls0, pls1, bl, bls0, bls1, n_epochs, batch_size, seed):
    """Train RL with greenhouse sim.

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

        if pl == 'mlp':
            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(pls0, pls1),
            )
        elif pl == 'lstm':
            policy = GaussianLSTMPolicy(
                env_spec=env.spec,
                hidden_dim=pls0,
            )

        if bl == 'Gaussian':
            baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            hidden_sizes=(bls0, bls1),
            use_trust_region=True)
        elif bl == 'Linear':
            baseline = LinearFeatureBaseline(env_spec=env.spec)
        
        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True,
                             worker_class=VecWorker,
                             worker_args=dict(n_envs=6),
                            #  n_workers=96
                             )

        #  LocalSample debug
        # sampler = LocalSampler(agents=policy,
        #                        envs=env,
        #                        max_episode_length=env.spec.max_episode_length,
        #                        is_tf_worker=True
        #                        )
        
        if alg == 'TRPO':
            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        sampler=sampler,
                        discount=1,
                        max_kl_step=clip)
        elif alg == 'PPO':
            # NOTE: make sure when setting entropy_method to 'max', set
            # center_adv to False and turn off policy gradient. See
            # tf.algos.NPO for detailed documentation.
            algo = PPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                sampler=sampler,
                discount=1,
                gae_lambda=0.95,
                lr_clip_range=clip,
                optimizer_args=dict(
                    batch_size=32,
                    max_optimization_epochs=10,
                ),
                stop_entropy_gradient=True,
                entropy_method='max',
                policy_ent_coeff=0.02,
                center_adv=False,
            )

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=batch_size, store_episodes=False)


rl_greenhouse()