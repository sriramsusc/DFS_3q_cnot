# ---------------------------------------------------------------------------
# robust_pdqn.py
# PureJaxRL-flavoured implementation of Robust PDQN
#
#  • Handles hybrid (discrete + continuous) actions à-la PDQN:
#       action  =  (d, θ_d)              # d ∈ {0,…,D-1},  θ_d ∈ ℝ^k
#  • Two coupled networks
#       – QNetwork        : Q(s, d, θ_d)
#       – ParamNetwork    : θ̂_d = π_θ(s, d)
#  • Robustness tricks
#       – Double-Q targets + soft-target update (τ)
#       – Parameter-space exploration noise (σ-decay)
#       – Auto-clipped gradients (Optax)
#       – Optional adversarial parameter perturbation hook
#  • Training loop identical in spirit to your DQN: vmapped env, Flashbax
#
#  NOTE: Replace the stub HybridGymnaxEnv below with your real env that
#        exposes .action_space() -> (Discrete(D), Box(low, high, (k,)))
# ---------------------------------------------------------------------------

import os, functools, jax, jax.numpy as jnp, optax, chex, flax.linen as nn
import flax, wandb, flashbax as fbx, gymnax
from flax.training.train_state import TrainState
import gymnax.environments.spaces as spaces
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper


# ----------------------------- 1. ENVIRONMENT ------------------------------ #
# If your env is already hybrid, delete this stub and import yours.
class HybridGymnaxEnv(gymnax.environments.environment.Environment):
    """Minimal hybrid-action env stub: (d, θ) → next_state."""
    # implement reset, step, action_space, observation_space …
    pass


# --------------------------- 2. NETWORK DEFINITIONS ------------------------ #
class QNetwork(nn.Module):
    """Q(s, d, θ) – takes concatenated [obs, one-hot(d), θ]."""
    hidden: int = 256
    n_discrete: int = 0

    @nn.compact
    def __call__(self, obs, disc, cont):
        # obs : (B, obs_dim)
        # disc: (B, n_discrete) one-hot
        # cont: (B, k)
        x = jnp.concatenate([obs, disc, cont], axis=-1)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.Dense(1)(x)               # Q-value scalar
        return x.squeeze(-1)             # (B,)


class ParamNetwork(nn.Module):
    """π_θ(s, d) – predicts θ for each discrete action."""
    hidden: int = 256
    n_discrete: int = 0
    cont_dim: int = 0

    @nn.compact
    def __call__(self, obs):
        # Return (B, n_discrete, cont_dim)
        x = nn.relu(nn.Dense(self.hidden)(obs))
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.Dense(self.n_discrete * self.cont_dim)(x)
        return x.reshape(*obs.shape[:-1], self.n_discrete, self.cont_dim)


# ----------------------------- 3. DATA STRUCTS ----------------------------- #
@chex.dataclass(frozen=True)
class TimeStep:
    obs:   chex.Array
    disc:  chex.Array  # int32
    cont:  chex.Array  # float32 (k-vector)
    reward: chex.Array
    done:   chex.Array


class PDQNTrainState(TrainState):
    target_q_params: flax.core.FrozenDict
    q_updates: int
    steps:     int
    param_params: flax.core.FrozenDict        # π parameters
    target_param_params: flax.core.FrozenDict # target π


# -------------------------- 4. TRAIN FUNCTION FACTORY ---------------------- #
def make_pdqn_train(config):

    # ---- env plumbing identical to your DQN -------------------------------- #
    basic_env, env_p = HybridGymnaxEnv(), None  # swap with real env
    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)

    n_D = env.action_space(env_p)[0].n
    k   = env.action_space(env_p)[1].shape[0]

    vmap_reset = lambda n: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n), env_p
    )
    vmap_step  = lambda n: lambda rng, st, disc, cont: jax.vmap(
        env.step, in_axes=(0, 0, 0, 0, None)
    )(jax.random.split(rng, n), st, disc, cont, env_p)

    def linear_lr(step):
        frac = 1.0 - (step / config["NUM_UPDATES"])
        return config["LR"] * frac

    # ------------------------------ inner loop ------------------------------ #
    def train(rng):

        # 4.1 initialise env, buffer ----------------------------------------- #
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        buffer = fbx.make_flat_buffer(
            max_length         = config["BUFFER_SIZE"],
            min_length         = config["BATCH_SIZE"],
            sample_batch_size  = config["BATCH_SIZE"],
            add_sequences      = False,
            add_batch_size     = config["NUM_ENVS"],
        )
        buffer = buffer.replace(                 # note: we call methods on *buffer*
            init        = jax.jit(buffer.init),
            add         = jax.jit(buffer.add,    donate_argnums=0),
            sample      = jax.jit(buffer.sample),
            can_sample  = jax.jit(buffer.can_sample),
        )
        dummy_step = TimeStep(
            obs=jnp.zeros(env.observation_space(env_p).shape),
            disc=jnp.zeros((), dtype=jnp.int32),
            cont=jnp.zeros((k,)),
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
        )
        buffer_state = buffer.init(dummy_step)

        # 4.2 networks & opt -------------------------------------------------- #
        q_net     = QNetwork(n_discrete=n_D)
        param_net = ParamNetwork(n_discrete=n_D, cont_dim=k)

        rng, qkey, pkey = jax.random.split(rng, 3)
        q_params = q_net.init(qkey,
                              jnp.zeros(env.observation_space(env_p).shape),
                              jax.nn.one_hot(0, n_D),
                              jnp.zeros((k,)))
        p_params = param_net.init(pkey, jnp.zeros(env.observation_space(env_p).shape))

        tx_q     = optax.chain(optax.clip_by_global_norm(10.0),  # grad clip
                               optax.adamw(learning_rate=linear_lr))
        tx_param = optax.chain(optax.clip_by_global_norm(10.0),
                               optax.adamw(learning_rate=linear_lr))

        state = PDQNTrainState.create(
            apply_fn=q_net.apply,
            params=q_params,
            tx=tx_q,
            target_q_params=jax.tree_util.tree_map(jnp.copy, q_params),
            q_updates=0,
            steps=0,
            param_params=p_params,
            target_param_params=jax.tree_util.tree_map(jnp.copy, p_params),
            # opt_state=tx_q.init(q_params),
        )
        opt_state        = state.opt_state
        opt_state_param = tx_param.init(p_params)

        # 4.3 exploration helpers ------------------------------------------- #
        def sample_theta(rng, theta_hat, sigma):
            noise = sigma * jax.random.normal(rng, theta_hat.shape)
            return jnp.clip(theta_hat + noise, -1.0, 1.0)  # clip to env box

        def select_discrete(rng, q_vals, eps):
            greedy = jnp.argmax(q_vals, axis=-1)
            rand   = jax.random.randint(rng, greedy.shape, 0, n_D)
            mask   = (jax.random.uniform(rng, greedy.shape) < eps)
            return jnp.where(mask, rand, greedy)

        # 4.4 single update step (jit-scanned) ------------------------------- #
        def _update(carry, _):
            st, buf_st, env_st, obs, rng = carry
            rng, rng_eps, rng_sig, rng_step = jax.random.split(rng, 4)
            eps  = jnp.maximum(config["EPS_FINISH"],
                               config["EPS_START"] - st.steps / config["EPS_DECAY"])
            sig  = jnp.maximum(config["SIGMA_END"],
                               config["SIGMA_START"] - st.steps / config["SIGMA_DECAY"])

            # -----  discrete + continuous policy evaluation ----------------- #
            theta_hat = param_net.apply(st.param_params, obs)        # (B, D, k)
            batch_size = obs.shape[0]
            def q_for_d(d):
                disc_mat = jnp.broadcast_to(
                    jax.nn.one_hot(d, n_D),           # (n_D,)
                    (batch_size, n_D)                 # (B, n_D)
                )
                return q_net.apply(
                    st.params,
                    obs,                              # (B, obs_dim)
                    disc_mat,                         # (B, n_D)
                    theta_hat[:, d]                   # (B, k)
                )                                     # → (B,)

            q_vals = jax.vmap(q_for_d)(jnp.arange(n_D))   # (D, B)
            q_vals = q_vals.T                              # (B, D) 

            disc_act  = select_discrete(rng_eps, q_vals, eps)        # (N,)
            cont_act  = sample_theta(rng_sig,
                                     jnp.take_along_axis(theta_hat,
                                                         disc_act[:,None,None],
                                                         axis=1).squeeze(1),
                                     sig)                             # (N,k)

            # -----  step env ------------------------------------------------- #
            obs_next, env_st, rew, done, info = vmap_step(config["NUM_ENVS"])(
                rng_step, env_st, disc_act, cont_act)

            st = st.replace(steps=st.steps + config["NUM_ENVS"])

            # -----  buffer add ---------------------------------------------- #
            ts = TimeStep(obs=obs,
                          disc=disc_act,
                          cont=cont_act,
                          reward=rew,
                          done=done)
            buf_st = buffer.add(buf_st, ts)

            # -----  learning phase ------------------------------------------ #
            def _learn(st, opt_q, opt_p, rng):
                batch = buffer.sample(buf_st, rng).experience
                s, a, θ, r, d, s2 = (batch.first.obs,
                                     batch.first.disc,
                                     batch.first.cont,
                                     batch.first.reward,
                                     batch.first.done,
                                     batch.second.obs)

                # ---------- target θ′ and Q′ (Double Q) -------------------- #
                θ2_hat   = param_net.apply(st.param_params, s2)  # (B,D,k)
                q2_eval  = jax.vmap(lambda d: q_net.apply(
                    st.params, s2, jax.nn.one_hot(d, n_D), θ2_hat[:,d]))(
                        jnp.arange(n_D)).T
                a2_star  = jnp.argmax(q2_eval, axis=1)                   # greedy by online net
                θ2_star  = jnp.take_along_axis(θ2_hat, a2_star[:,None,None], axis=1
                                   ).squeeze(1)
                q2_target = q_net.apply(st.target_q_params, s2,
                                        jax.nn.one_hot(a2_star, n_D), θ2_star)
                y = r + (1.0 - d) * config["GAMMA"] * q2_target

                # ---------- Q loss ----------------------------------------- #
                def q_loss(params):
                    q_pred = q_net.apply(params, s,
                                         jax.nn.one_hot(a, n_D), θ)
                    return jnp.square(q_pred - y).mean()

                q_loss_val, q_grads = jax.value_and_grad(q_loss)(st.params)
                updates_q, opt_q = tx_q.update(q_grads, opt_q,
                                               params=st.params)
                st_q_params = optax.apply_updates(st.params, updates_q)

                # ---------- π loss (maximise Q wrt θ) ---------------------- #
                def p_loss(p_params):
                    θ_hat_sa = jnp.take_along_axis(
                        param_net.apply(p_params, s), a[:,None,None], axis=1
                    ).squeeze(1)
                    q_val = q_net.apply(st.params, s,
                                        jax.nn.one_hot(a, n_D), θ_hat_sa)
                    return (-q_val).mean()

                p_loss_val, p_grads = jax.value_and_grad(p_loss)(st.param_params)
                updates_p, opt_p = tx_param.update(p_grads, opt_p,
                                                   params=st.param_params)
                st_p_params = optax.apply_updates(st.param_params, updates_p)

                st = st.replace(params=st_q_params,
                                param_params=st_p_params,
                                q_updates=st.q_updates + 1)

                metrics = dict(q_loss=q_loss_val,
                               p_loss=p_loss_val,
                               return_mean=info["returned_episode_returns"].mean(),
                               steps=st.steps,
                               updates=st.q_updates)
                return st, opt_q, opt_p, metrics

            rng, rng_learn = jax.random.split(rng)
            st, opt_state, opt_state_param, metrics = jax.lax.cond(
                (buffer.can_sample(buf_st)
                 & (st.steps > config["LEARN_STARTS"])
                 & (st.steps % config["TRAIN_INTERVAL"] == 0)),
                _learn,
                lambda st, opt_q, opt_p, rng: (st, opt_q, opt_p,
                                               dict(q_loss=0.0, p_loss=0.0,
                                                    return_mean=0.0,
                                                    steps=st.steps,
                                                    updates=st.q_updates)),
                st, opt_state, opt_state_param, rng_learn
            )

            # -----  target updates ---------------------------------------- #
            def soft_upd(target, src):
                return optax.incremental_update(src, target, config["TAU"])

            st = st.replace(
                target_q_params=jax.lax.cond(
                    st.q_updates % config["TARGET_INTERVAL"] == 0,
                    lambda _: soft_upd(st.target_q_params, st.params),
                    lambda _: st.target_q_params,
                    operand=None),
                target_param_params=jax.lax.cond(
                    st.q_updates % config["TARGET_INTERVAL"] == 0,
                    lambda _: soft_upd(st.target_param_params, st.param_params),
                    lambda _: st.target_param_params,
                    operand=None),
            )

            # -----  wandb logging ----------------------------------------- #
            if config["WANDB_MODE"] == "online":
                jax.debug.callback(
                    lambda m: wandb.log(m) if m["steps"] % 100 == 0 else None,
                    metrics
                )

            return (st, buf_st, env_st, obs_next, rng), metrics

        # 4.5 run scan ------------------------------------------------------- #
        rng, run_key = jax.random.split(rng)
        carry = (state, buffer_state, env_state, init_obs, run_key)
        carry, metrics = jax.lax.scan(
            _update, carry, None, config["NUM_UPDATES"]
        )
        return dict(carry=carry, metrics=metrics)

    return train


# --------------------------- 5. DRIVER (example) --------------------------- #
def main():
    cfg = dict(
        NUM_ENVS=10,
        BUFFER_SIZE=50_000,
        BATCH_SIZE=256,
        TOTAL_TIMESTEPS=5e5,
        NUM_UPDATES= int(5e5 // 10),
        GAMMA=0.99,
        LR=5e-4,
        EPS_START=1.0, EPS_FINISH=0.05, EPS_DECAY=3e5,
        SIGMA_START=0.2, SIGMA_END=0.02, SIGMA_DECAY=3e5,
        TARGET_INTERVAL=500,
        LEARN_STARTS=5_000,
        TRAIN_INTERVAL=1,
        TAU=0.005,
        WANDB_MODE="disabled",
    )

    wandb.init(project="robust_pdqn", mode=cfg["WANDB_MODE"], config=cfg,
               tags=["PDQN", "jax"])
    rng = jax.random.PRNGKey(0)
    train_fn = jax.jit(jax.vmap(make_pdqn_train(cfg)))
    _ = jax.block_until_ready(train_fn(jax.random.split(rng, 1)))


if __name__ == "__main__":
    main()
