from easydict import EasyDict as edict
from ding.config import compile_config

main_config = edict({
    "exp_name": "pdqn_exchange_cnot",

    
    # ────────────────────── environment ────────────────────── #
    "env": {
        "import_names": ["exch_gym_env"],
        "type": "ExchangeCNOTEnvDI",
        "max_episode_steps": 18,
        "collector_env_num": 1,
        "evaluator_env_num": 1,
        "use_act_scale": True,
    },

    # ───────────────────────── policy ───────────────────────── #
    "policy": {
        "type": "pdqn_command",
        "cuda": True,  # use GPU for training
        # ‣ model description → **one** dict for both branches
        "model": {
            "obs_shape": 168,
            "action_shape": edict({
                "action_type_shape": 5,   # discrete: 5 neighbour pairs
                "action_args_shape": 1,   # continuous: swap-power p
                "encoder_hidden_size_list": [256, 256, 256]
            }),
        },

        # ‣ learning hyper-params
        "learn": {
            "multi_gpu": False,
            "hook": {"load_on_driver": True},
            "train_epoch": 100,
            "batch_size": 64,

            # ──► PDQN needs these two ◄──
            "learning_rate_dis": 1e-3,   # discrete Q-network
            "learning_rate_cont": 1e-3,  # continuous Q-network
            "update_circle": 10,
            "weight_decay": 0,
            
        },
        # ‣ data collection / evaluation
        "collect": {
            "n_sample": 320,
            # "unroll_len": 3,
            "noise": True,
            # NEW – Gaussian with σ=0.7 mapped to [-2,2]
            "noise_sigma": 0.7,
        },
        "eval":    {"evaluator": {"eval_freq": 10, "n_episode": 5}},

        # ‣ misc
        "other": {
            "eps": {
                "type": "exp",
                "start": 1.0,
                "end": 0.05,
                "decay": 10000,
            },
            "replay_buffer": {"replay_buffer_size": 100_000},
        },
    },
})

# create_cfg now includes the minimal pieces DI-engine needs
create_config = edict({
    # 1. env_manager key so compile_config won't crash
    "env_manager": {
        "type": "base",      # matches your main_config.manager
    },
    # 2. env must point to your registered class
    "env": {
        "import_names": ["exch_gym_env"],
        "type": "ExchangeCNOTEnvDI",
    },
    # 3. policy command name
    "policy": {
        "type": "pdqn",
    },
})