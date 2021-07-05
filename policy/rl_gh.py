#!/usr/bin/env python3
"""This is a script to train a greenhouse sim with RL algorithm.
"""
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../model_forward/')
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
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer

@click.command()
@click.option('--seed', default=1)
@click.option('--n_epochs', default=100)
@click.option('--batch_size', default=4000)
@click.option('--plot', default=False)
@wrap_experiment
def rl_greenhouse(ctxt, seed, n_epochs, batch_size, plot):
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

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        # baseline = LinearFeatureBaseline(env_spec=env.spec)
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            use_trust_region=True,
        )

        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True,
                               worker_class=DefaultWorker,
                            #    worker_class=VecWorker,
                            #    worker_args=dict(n_envs=12),
                               n_workers=1
                               )
        # sampler = RaySampler(agents=policy,
        #                      envs=env,
        #                      max_episode_length=env.spec.max_episode_length,
        #                      is_tf_worker=True)
        
        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    sampler=sampler,
                    discount=1,
                    max_kl_step=0.01)

        # # NOTE: make sure when setting entropy_method to 'max', set
        # # center_adv to False and turn off policy gradient. See
        # # tf.algos.NPO for detailed documentation.
        # algo = PPO(
        #     env_spec=env.spec,
        #     policy=policy,
        #     baseline=baseline,
        #     sampler=sampler,
        #     discount=1,
        #     gae_lambda=0.95,
        #     lr_clip_range=0.2,
        #     optimizer_args=dict(
        #         batch_size=32,
        #         max_optimization_epochs=10,
        #     ),
        #     stop_entropy_gradient=True,
        #     entropy_method='max',
        #     policy_ent_coeff=0.02,
        #     center_adv=False,
        # )

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=batch_size, plot=plot)


rl_greenhouse()