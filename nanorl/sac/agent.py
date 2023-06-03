"""Soft Actor-Critic implementation."""

from dataclasses import dataclass

from functools import partial
from typing import Any, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
import trax.models.transformer

from nanorl import agent
from nanorl.distributions import TanhNormal
from nanorl.networks import MLP, Ensemble, StateActionValue, TransformerAnd, subsample_ensemble
from nanorl.specs import EnvironmentSpec, zeros_like
from nanorl.types import LogDict, Transition


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda _: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


# @partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(
    rng, apply_fn, params, observations: np.ndarray
) -> tuple[jnp.ndarray, Any]:
    key, rng = jax.random.split(rng)
    print("observations: ", observations)
    dist = jnp.stack([apply_fn({"params": params}, observation) for observation in observations])
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> jnp.ndarray:
    dist = apply_fn({"params": params}, observations)
    return dist.mode()


@dataclass(frozen=True)
class SACConfig:
    """Configuration options for SAC."""

    num_qs: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    hidden_dims: Sequence[int] = (256, 256, 256)
    activation: str = "gelu"
    num_min_qs: Optional[int] = None
    critic_dropout_rate: Optional[float] = None
    critic_layer_norm: bool = False
    tau: float = 0.005
    target_entropy: Optional[float] = None
    init_temperature: float = 1.0
    backup_entropy: bool = True
    critic_utd_ratio: int = 1
    actor_utd_ratio: int = 1
    use_transformer: bool = False


class SAC(agent.Agent):
    """Soft-Actor Critic (SAC)."""

    actor: TrainState
    rng: Any
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float = struct.field(pytree_node=False)
    discount: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    critic_utd_ratio: int = struct.field(pytree_node=False)
    actor_utd_ratio: int = struct.field(pytree_node=False)
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)

    @staticmethod
    def initialize(
        spec: EnvironmentSpec,
        config: SACConfig,
        seed: int = 0,
        discount: float = 0.99,
    ) -> "SAC":
        """Initializes the agent from the given environment spec and config."""

        action_dim = spec.action.shape[-1]
        observations = zeros_like(spec.observation)
        actions = zeros_like(spec.action)

        if config.target_entropy is None:
            target_entropy = -0.5 * action_dim

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        if config.use_transformer:
            actor_base_cls = partial(
                TransformerAnd,
                hidden_dim=config.hidden_dims[0],
                n_layers=3,
            )
        else:
            actor_base_cls = partial(
                MLP,
                hidden_dims=config.hidden_dims,
                activation=getattr(nn, config.activation),
                activate_final=True,
            )
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=config.actor_lr),
        )

        critic_base_cls = partial(
            MLP,
            hidden_dims=config.hidden_dims,
            activation=getattr(nn, config.activation),
            activate_final=True,
            dropout_rate=config.critic_dropout_rate,
            use_layer_norm=config.critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=config.num_qs)
        observations_flattened = np.concatenate([np.atleast_1d(observation.ravel()) for observation in observations.values()])
        critic_params = critic_def.init(critic_key, observations_flattened, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=config.critic_lr),
        )
        target_critic_def = Ensemble(critic_cls, num=config.num_min_qs or config.num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(config.init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=config.temp_lr),
        )

        return SAC(
            actor=actor,
            rng=rng,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=config.tau,
            discount=discount,
            num_qs=config.num_qs,
            num_min_qs=config.num_min_qs,
            backup_entropy=config.backup_entropy,
            critic_utd_ratio=config.critic_utd_ratio,
            actor_utd_ratio=config.actor_utd_ratio,
        )

    def update_actor(self, transitions: Transition) -> tuple["SAC", LogDict]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params):
            dist = self.actor.apply_fn(
                {"params": actor_params}, transitions.observation
            )
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                transitions.observation,
                actions,
                True,  # training.
                rngs={"dropout": key2},
            )
            q = qs.mean(axis=0)
            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> tuple["SAC", LogDict]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(self, transitions: Transition) -> tuple["SAC", LogDict]:
        dist = self.actor.apply_fn(
            {"params": self.actor.params}, transitions.next_observation
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key=key,
            params=self.target_critic.params,
            num_sample=self.num_min_qs,
            num_qs=self.num_qs,
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            transitions.next_observation,
            next_actions,
            True,  # training.
            rngs={"dropout": key},
        )
        next_q = next_qs.min(axis=0)

        target_q = transitions.reward + self.discount * transitions.discount * next_q

        if self.backup_entropy:
            target_q -= (
                self.discount
                * transitions.discount
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params):
            qs = self.critic.apply_fn(
                {"params": critic_params},
                transitions.observation,
                transitions.action,
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @jax.jit
    def update(self, transitions: Transition) -> tuple["SAC", LogDict]:
        new_agent = self

        # Update critic.
        for i in range(self.critic_utd_ratio):

            def slice(x):
                batch_size = x.shape[0] // self.critic_utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_transition = jax.tree_util.tree_map(slice, transitions)
            new_agent, critic_info = new_agent.update_critic(mini_transition)

        # Update actor.
        for i in range(self.actor_utd_ratio):

            def slice(x):
                batch_size = x.shape[0] // self.actor_utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_transition = jax.tree_util.tree_map(slice, transitions)
            new_agent, actor_info = new_agent.update_actor(mini_transition)

        # Update temperature.
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        return new_agent, {**actor_info, **critic_info, **temp_info}

    def sample_actions(self, observations: np.ndarray) -> tuple["SAC", np.ndarray]:
        actions, new_rng = _sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations
        )
        return self.replace(rng=new_rng), np.asarray(actions)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions)
