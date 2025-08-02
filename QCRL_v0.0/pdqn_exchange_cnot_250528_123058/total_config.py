exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 1,
            'retry_type': 'reset',
            'auto_reset': True,
            'step_timeout': None,
            'reset_timeout': None,
            'retry_waiting_time': 0.1,
            'cfg_type': 'BaseEnvManagerDict',
            'type': 'base'
        },
        'stop_value': 10000000000,
        'n_evaluator_episode': 4,
        'import_names': ['exch_gym_env'],
        'type': 'ExchangeCNOTEnvDI',
        'max_episode_steps': 18,
        'collector_env_num': 8,
        'evaluator_env_num': 3,
        'use_act_scale': True
    },
    'policy': {
        'model': {
            'obs_shape': 163,
            'action_shape': {
                'action_type_shape': 5,
                'action_args_shape': 1
            }
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'log_policy': True,
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 10000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'resume_training': False,
            'update_per_collect': 3,
            'batch_size': 64,
            'learning_rate': 0.001,
            'target_theta': 0.005,
            'ignore_done': False,
            'multi_gpu': False,
            'hook': {
                'load_on_driver': True
            },
            'train_epoch': 100,
            'learning_rate_dis': 0.001,
            'learning_rate_cont': 0.001,
            'update_circle': 10,
            'weight_decay': 0
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'cfg_type': 'SampleSerialCollectorDict',
                'type': 'sample'
            },
            'unroll_len': 1,
            'noise_sigma': 0.1,
            'n_sample': 320,
            'noise': True,
            'action_args_noise': {
                'type': 'normal',
                'sigma': 0.7
            }
        },
        'eval': {
            'evaluator': {
                'eval_freq': 1000,
                'render': {
                    'render_freq': -1,
                    'mode': 'train_iter'
                },
                'figure_path': None,
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'n_episode': 4,
                'stop_value': 10000000000
            }
        },
        'other': {
            'replay_buffer': {
                'type': 'advanced',
                'replay_buffer_size': 100000,
                'max_use': float("inf"),
                'max_staleness': float("inf"),
                'alpha': 0.6,
                'beta': 0.4,
                'anneal_step': 100000,
                'enable_track_used_data': False,
                'deepcopy': False,
                'thruput_controller': {
                    'push_sample_rate_limit': {
                        'max': float("inf"),
                        'min': 0
                    },
                    'window_seconds': 30,
                    'sample_min_limit_ratio': 1
                },
                'monitor': {
                    'sampled_data_attr': {
                        'average_range': 5,
                        'print_freq': 200
                    },
                    'periodic_thruput': {
                        'seconds': 60
                    }
                },
                'cfg_type': 'AdvancedReplayBufferDict'
            },
            'eps': {
                'type': 'exp',
                'start': 1.0,
                'end': 0.05,
                'decay': 10000
            },
            'commander': {
                'cfg_type': 'BaseSerialCommanderDict'
            }
        },
        'on_policy': False,
        'cuda': True,
        'multi_gpu': False,
        'bp_update_sync': True,
        'traj_len_inf': False,
        'type': 'pdqn_command',
        'priority': False,
        'priority_IS_weight': False,
        'discount_factor': 0.97,
        'nstep': 1,
        'cfg_type': 'PDQNCommandModePolicyDict'
    },
    'exp_name': 'pdqn_exchange_cnot_250528_123058',
    'seed': 42
}
