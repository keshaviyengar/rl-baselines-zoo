import argparse
import difflib
import os
from collections import OrderedDict
from pprint import pprint
import warnings
import importlib

# For pybullet envs
warnings.filterwarnings("ignore")
import gym

import numpy as np
import yaml

from mpi4py import MPI

from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise

from utils import make_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class
from utils.hyperparams_opt import hyperparam_optimization
from utils.noise import LinearNormalActionNoise

from stable_baselines.her.utils import HERGoalEnvWrapper

import ctm2_envs
import ctr_envs
import ctm_envs
from utils.callback_object import CallbackObject
from utils.ctm_callback import CtmCallback

from stable_baselines.logger import configure

# TODO: Move experiment arguments to yaml, rather than it being here.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, nargs='+', default=["CartPole-v1"], help='environment ID(s)')
    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                        default='', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                        type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,
                        type=int)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--n-trials', help='Number of trials for optimizing hyperparameters', type=int, default=10)
    parser.add_argument('-optimize', '--optimize-hyperparameters', action='store_true', default=False,
                        help='Run hyperparameters search')
    parser.add_argument('--n-jobs', help='Number of parallel jobs when optimizing hyperparameters', type=int, default=1)
    parser.add_argument('--sampler', help='Sampler to use when optimizing hyperparameters', type=str,
                        default='skopt', choices=['random', 'tpe', 'skopt'])
    parser.add_argument('--pruner', help='Pruner to use when optimizing hyperparameters', type=str,
                        default='none', choices=['halving', 'median', 'none'])
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[],
                        help='Additional external Gym environment package modules to import (e.g. gym_minigrid)')
    parser.add_argument('--load-weights-file',
                        help='Load the policy weights and shared weights from a file.',
                        default='', type=str)
    parser.add_argument('--load-weights-env',
                        help='Choose the environment-id from which the weights file is loaded. Incorrect id will cause error.',
                        default='', type=str)
    parser.add_argument('--render-type', help='Choose a rendering type during evaluation: empty, record or human',
                        default='', type=str)
    parser.add_argument('--noise-experiment-id',
                        help='Choose the experiment number. Refer to the excel sheet or comments in code.',
                        default=0, type=int)
    parser.add_argument('--goal-tolerance-experiment-id',
                        help='Choose the experiment number for exact,'
                             'curriculum experiments. 1: exp decay, 2: linear, 3: chi-squared, 4: constant',
                        default=0, type=int)
    parser.add_argument('--joint-representation', help='joint representation', type=str, default='trig')
    parser.add_argument('--relative-q', help='relative or absolute joint', type=bool, default=False)
    parser.add_argument('--inc-goals-obs', help='include achieved and desired goal in observation', type=bool,
                        default=True)
    args = parser.parse_args()

    # Set log directory
    os.environ["OPENAI_LOG_FORMAT"] = 'stdout,log,csv'
    os.environ["OPENAI_LOGDIR"] = args.log_folder
    configure()

    env_ids = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())

    for env_id in env_ids:
        # If the environment is not found, suggest the closest match
        if env_id not in registered_envs:
            try:
                closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
            except IndexError:
                closest_match = "'no close match found...'"
            raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

    set_global_seeds(args.seed)

    if args.trained_agent != "":
        assert args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent), \
            "The trained_agent must be a valid path to a .pkl file"

    rank = 0
    if MPI.COMM_WORLD.Get_size() > 1:
        print("Using MPI for multiprocessing with {} workers".format(MPI.COMM_WORLD.Get_size()))
        rank = MPI.COMM_WORLD.Get_rank()
        print("Worker rank: {}".format(rank))

        args.seed += rank
        if rank != 0:
            args.verbose = 0
            args.tensorboard_log = ''

    for env_id in env_ids:
        tensorboard_log = None if args.tensorboard_log == '' else os.path.join(args.tensorboard_log, env_id)

        is_atari = False
        if 'NoFrameskip' in env_id:
            is_atari = True

        print("=" * 10, env_id, "=" * 10)

        # Load hyperparameters from yaml file
        with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
            hyperparams_dict = yaml.load(f)
            if env_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[env_id]
            elif is_atari:
                hyperparams = hyperparams_dict['atari']
            else:
                raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))

        # Noise experiments
        # 1-5: 2 Tube, 6-10: 3 Tube, 11-15: 4 Tube
        # 1,6,11: Gaussian noise
        # 2,7,12: Multivariate Gaussian noise
        # 3,8,13: Two tube optimized noise
        # 4,9,14: Parameter noise
        # 5,10,15: OU Noise
        if args.noise_experiment_id == 1:
            print("two tube gaussian noise 0.35 std")
            hyperparams['noise_type'] = 'normal'
            hyperparams['noise_std'] = 0.35
        elif args.noise_experiment_id == 2:
            print("two tube gaussian seperate noise 0.025, 0.00065 std")
            hyperparams['noise_type'] = 'normal'
            hyperparams['noise_std'] = [0.00065, 0.00065, 0.025, 0.025]
        elif args.noise_experiment_id == 3:
            print("two tube varied gaussian noise 0.0009, 0.0004 std")
            hyperparams['noise_type'] = 'normal'
            hyperparams['noise_std'] = [0.0009, 0.0004, 0.025, 0.025]
        elif args.noise_experiment_id == 4:
            print("two tube parameter noise 0.24 std")
            hyperparams['noise_type'] = 'adaptive-param'
            hyperparams['noise_std'] = 0.24
        elif args.noise_experiment_id == 5:
            print("two tube OU noise")
            hyperparams['noise_type'] = 'ornstein-uhlenbeck'
            hyperparams['noise_std'] = [0.00065, 0.00065, 0.025, 0.025]
            hyperparams['theta'] = 0.3
            hyperparams['noise_mean'] = [-0.001, -0.001, 0, 0]
        elif args.noise_experiment_id == 6:
            print("three tube gaussian noise 0.35 std")
            hyperparams['noise_type'] = 'normal'
            hyperparams['noise_std'] = 0.35
        elif args.noise_experiment_id == 7:
            print("three tube gaussian noise 0.025, 0.00065 std")
            hyperparams['noise_type'] = 'normal'
            hyperparams['noise_std'] = [0.00065, 0.00065, 0.00065, 0.025, 0.025, 0.025]
        elif args.noise_experiment_id == 8:
            print("three tube varied gaussian noise 0.0009, 0.0009, 0.0004 std")
            hyperparams['noise_type'] = 'normal'
            hyperparams['noise_std'] = [0.0009, 0.009, 0.0004, 0.025, 0.025, 0.025]
        elif args.noise_experiment_id == 9:
            print("three tube parameter noise 0.6771411441114241 std")
            hyperparams['noise_type'] = 'adaptive-param'
            # hyperparams['noise_std'] = 0.24
            hyperparams['noise_std'] = 0.6771411441114241
        elif args.noise_experiment_id == 10:
            print("three tube OU noise")
            hyperparams['noise_type'] = 'ornstein-uhlenbeck'
            hyperparams['noise_std'] = [0.00065, 0.00065, 0.00065, 0.025, 0.025, 0.025]
            hyperparams['theta'] = 0.3
            hyperparams['noise_mean'] = [-0.001, -0.001, -0.001, 0, 0, 0]
        elif args.noise_experiment_id == 11:
            print("four tube gaussian noise 0.35 std")
            hyperparams['noise_type'] = 'normal'
            hyperparams['noise_std'] = 0.35
        elif args.noise_experiment_id == 12:
            print("four tube gaussian noise 0.025, 0.00065 std")
            hyperparams['noise_type'] = 'normal'
            hyperparams['noise_std'] = [0.025, 0.00065, 0.025, 0.00065, 0.025, 0.00065, 0.025, 0.00065]
        elif args.noise_experiment_id == 13:
            print("four tube varied gaussian noise 0.0009, 0.0009, 0.0009, 0.0004 std")
            hyperparams['noise_type'] = 'normal'
            hyperparams['noise_std'] = [0.025, 0.0009, 0.025, 0.0009, 0.025, 0.0009, 0.025, 0.0009]
        elif args.noise_experiment_id == 14:
            print("four tube parameter noise 0.24 std")
            hyperparams['noise_type'] = 'adaptive-param'
            hyperparams['noise_std'] = 0.24
        elif args.noise_experiment_id == 15:
            print("four tube OU noise")
            hyperparams['noise_type'] = 'ornstein-uhlenbeck'
        else:
            print("Non experiment id, custom noise selection.")

        if len(args.joint_representation) is not 0:
            hyperparams['env_kwargs']['joint_representation'] = args.joint_representation
            print(args.joint_representation)

        if args.relative_q is not None:
            hyperparams['env_kwargs']['relative_q'] = args.relative_q
            print('relative q: ', args.relative_q)

        if args.relative_q is not None:
            hyperparams['env_kwargs']['inc_goals_obs'] = args.inc_goals_obs
            print('inc_goals_obs: ', args.inc_goals_obs)

        # Sort hyperparams that will be saved
        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

        algo_ = args.algo
        # HER is only a wrapper around an algo
        if args.algo == 'her':
            algo_ = saved_hyperparams['model_class']
            assert algo_ in {'sac', 'ddpg', 'dqn', 'td3'}, "{} is not compatible with HER".format(algo_)
            # Retrieve the model class
            hyperparams['model_class'] = ALGOS[saved_hyperparams['model_class']]

        if args.verbose > 0:
            pprint(saved_hyperparams)
        n_envs = hyperparams.get('n_envs', 1)

        if args.verbose > 0:
            print("Using {} environments".format(n_envs))

        # Should we overwrite the number of timesteps?
        if args.n_timesteps > 0:
            if args.verbose:
                print("Overwriting n_timesteps with n={}".format(args.n_timesteps))
            n_timesteps = args.n_timesteps
        else:
            n_timesteps = int(hyperparams['n_timesteps'])

        normalize = False
        normalize_kwargs = {}
        if 'normalize' in hyperparams.keys():
            normalize = hyperparams['normalize']
            if isinstance(normalize, str):
                normalize_kwargs = eval(normalize)
                normalize = True
            del hyperparams['normalize']

        if 'policy_kwargs' in hyperparams.keys():
            hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

        # Delete keys so the dict can be pass to the model constructor
        if 'n_envs' in hyperparams.keys():
            del hyperparams['n_envs']
        if 'evaluation' in hyperparams.keys():
            del hyperparams['evaluation']
        del hyperparams['n_timesteps']

        # obtain a class object from a wrapper name string in hyperparams
        # and delete the entry
        env_wrapper = get_wrapper_class(hyperparams)
        if 'env_wrapper' in hyperparams.keys():
            del hyperparams['env_wrapper']


        def create_env(n_envs):
            """
            Create the environment and wrap it if necessary
            :param n_envs: (int)
            :return: (gym.Env)
            """
            global hyperparams
            env_kwargs = hyperparams['env_kwargs']

            if is_atari:
                if args.verbose > 0:
                    print("Using Atari wrapper")
                env = make_atari_env(env_id, num_env=n_envs, seed=args.seed)
                # Frame-stacking with 4 frames
                env = VecFrameStack(env, n_stack=4)
            elif algo_ in ['dqn', 'ddpg']:
                if hyperparams.get('normalize', False):
                    print("WARNING: normalization not supported yet for DDPG/DQN")
                env = gym.make(env_id, **env_kwargs)
                env.seed(args.seed)
                if env_wrapper is not None:
                    env = env_wrapper(env)
            else:
                if n_envs == 1:
                    env = DummyVecEnv([make_env(env_id, 0, args.seed, wrapper_class=env_wrapper)])
                else:
                    # env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])
                    # On most env, SubprocVecEnv does not help and is quite memory hungry
                    env = DummyVecEnv(
                        [make_env(env_id, i, args.seed, wrapper_class=env_wrapper) for i in range(n_envs)])
                if normalize:
                    if args.verbose > 0:
                        if len(normalize_kwargs) > 0:
                            print("Normalization activated: {}".format(normalize_kwargs))
                        else:
                            print("Normalizing input and reward")
                    env = VecNormalize(env, **normalize_kwargs)
            # Optional Frame-stacking
            if hyperparams.get('frame_stack', False):
                n_stack = hyperparams['frame_stack']
                env = VecFrameStack(env, n_stack)
                print("Stacking {} frames".format(n_stack))
                del hyperparams['frame_stack']
            return env


        env = create_env(n_envs)
        env_kwargs = hyperparams['env_kwargs']
        eval_env = None
        if algo_ == 'her':
            if args.render_type != '':
                eval_env = HERGoalEnvWrapper(gym.make(env_id, **env_kwargs))
            else:
                eval_env = HERGoalEnvWrapper(
                    gym.make(env_id, **env_kwargs))
        if algo_ in ['ppo2', 'ddpg']:
                eval_env = gym.make(env_id, **env_kwargs)
        del hyperparams['env_kwargs']

        # Stop env processes to free memory
        if args.optimize_hyperparameters and n_envs > 1:
            env.close()

        # Parse noise string for DDPG and SAC
        if algo_ in ['ddpg', 'sac', 'td3'] and hyperparams.get('noise_type') is not None:
            noise_type = hyperparams['noise_type'].strip()
            noise_std = hyperparams['noise_std']
            n_actions = env.action_space.shape[0]
            if 'adaptive-param' in noise_type:
                print("ALGO: ", algo_)
                assert algo_ == 'ddpg', 'Parameter is not supported by SAC'
                hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                                    desired_action_stddev=noise_std)
            elif 'normal' in noise_type:
                if 'lin' in noise_type:
                    hyperparams['action_noise'] = LinearNormalActionNoise(mean=np.zeros(n_actions),
                                                                          sigma=noise_std * np.ones(n_actions),
                                                                          final_sigma=hyperparams.get('noise_std_final',
                                                                                                      0.0) * np.ones(
                                                                              n_actions),
                                                                          max_steps=n_timesteps)
                else:
                    hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                                    sigma=noise_std * np.ones(n_actions))
            elif 'ornstein-uhlenbeck' in noise_type:
                hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(initial_noise=np.zeros(n_actions),
                                                                           sigma=hyperparams.get('noise_std',
                                                                                                 0) * np.ones(
                                                                               n_actions),
                                                                           theta=hyperparams.get('theta', 0), dt=1,
                                                                           mean=hyperparams.get('noise_mean',
                                                                                                0) * np.ones(
                                                                               n_actions))
            else:
                raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
            print("Applying {} noise with std {}".format(noise_type, noise_std))
            del hyperparams['noise_type']
            del hyperparams['noise_std']
            if 'theta' in hyperparams:
                del hyperparams['theta']
            if 'noise_mean' in hyperparams:
                del hyperparams['noise_mean']
            if 'noise_std_final' in hyperparams:
                del hyperparams['noise_std_final']

        if args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent):
            # Continue training
            print("Loading pretrained agent")
            # Policy should not be changed
            del hyperparams['policy']

            model = ALGOS[args.algo].load(args.trained_agent, env=env, eval_env=eval_env,
                                          tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams)

            exp_folder = args.trained_agent.split('.pkl')[0]
            if normalize:
                print("Loading saved running average")
                env.load_running_average(exp_folder)

        elif args.optimize_hyperparameters:
            if args.verbose > 0:
                print("Optimizing hyperparameters")


            def create_model(*_args, **kwargs):
                """
                Helper to create a model with different hyperparameters

                """
                return ALGOS[args.algo](env=create_env(n_envs), tensorboard_log=tensorboard_log,
                                        verbose=0, **kwargs)


            data_frame = hyperparam_optimization(args.algo, create_model, create_env, n_trials=args.n_trials,
                                                 n_timesteps=n_timesteps, hyperparams=hyperparams,
                                                 n_jobs=args.n_jobs, seed=args.seed,
                                                 sampler_method=args.sampler, pruner_method=args.pruner,
                                                 verbose=args.verbose)

            report_name = "report_{}_{}-trials-{}-{}-{}.csv".format(env_id, args.n_trials, n_timesteps,
                                                                    args.sampler, args.pruner)

            log_path = os.path.join(args.log_folder, args.algo, report_name)

            if args.verbose:
                print("Writing report to {}".format(log_path))

            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            data_frame.to_csv(log_path)
            exit()
        else:
            # Train an agent from scratch
            if algo_ == 'ddpg':
                model = ALGOS[args.algo](env=env, eval_env=eval_env, tensorboard_log=tensorboard_log,
                                         verbose=args.verbose, **hyperparams)
            else:
                model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams)

        kwargs = {}
        if args.log_interval > -1:
            kwargs['log_interval'] = args.log_interval

        if algo_ in ['her']:
            obs_dim = eval_env.env.get_obs_dim()
            goal_dim = eval_env.env.get_goal_dim()
            callback_object = CtmCallback(args.log_folder, algo_, n_timesteps, args.inc_goals_obs, obs_dim, goal_dim)
            kwargs['callback'] = callback_object.callback

        # Load an experiments .pkl network weights if needed
        if not args.load_weights_env == '':
            old_model = ALGOS[args.algo].load(args.load_weights_file, env=gym.make(args.load_weights_env))
            old_model_params = old_model.model.get_parameters()
            old_model_params = dict(
                (key, value) for key, value in old_model_params.items() if ("model/" in key or "target/" in key))
            model.model.load_parameters(old_model_params, exact_match=False)

        # Set saving paths before training and save the hyperparameters
        log_path = "{}/".format(args.log_folder)
        print('log path: ', log_path)
        save_path = os.path.join(log_path, "{}_{}".format(env_id, get_latest_run_id(log_path, env_id) + 1))
        print('save path: ', save_path)
        if rank == 0:
            # os.makedirs(save_path, exist_ok=True)
            # Save hyperparams
            with open(os.path.join(log_path, 'config.yml'), 'w') as f:
                yaml.dump(saved_hyperparams, f)

        model.learn(n_timesteps, **kwargs)

        # Only save worker of rank 0 when using mpi
        print('rank: ', rank)
        if rank == 0:
            print("Saving to {}".format(log_path))
            model.save("{}/{}".format(log_path, env_id))
