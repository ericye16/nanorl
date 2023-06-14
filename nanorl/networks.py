"""Neural network module."""

from typing import Callable, Optional, Sequence, Type

import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers import BertConfig
from transformers.models.bert.modeling_flax_bert import FlaxBertEncoder

from nanorl import types

default_init = nn.initializers.xavier_uniform


class TransformerAnd(nn.Module):
    config: BertConfig
    sequence_len: int
    dtype: jnp.dtype = jnp.float32
    obs_ts_dim: int = 89
    fingering_dim: int = 10

    def setup(self):
        self.position_embeddings = nn.Embed(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.encoder = FlaxBertEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=False,
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.note_conv = nn.Conv(features=self.config.hidden_size, kernel_size=(10,), strides=10)
        self.fingering_conv = nn.Conv(features=self.config.hidden_size, kernel_size=(10,), strides=10)

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        goal_states = x['goal']
        fingering_states = x['fingering']
        if len(x['reward'].shape) < len(goal_states.shape):
            x['reward'] = jnp.expand_dims(x['reward'], -1)
        fixed_state = jnp.concatenate([
            v
            for k, v in x.items()
            if k not in ('goal', 'fingering')
        ], axis=-1)

        if len(goal_states.shape) == 1:
            goal_states = jnp.expand_dims(goal_states, 0)
            fingering_states = jnp.expand_dims(fingering_states, 0)
            fixed_state = jnp.expand_dims(fixed_state, 0)

        goal_reshaper = lambda x: jnp.reshape(x, (self.sequence_len, self.obs_ts_dim))
        goal_states = jax.vmap(goal_reshaper)(goal_states)
        fingering_reshaper = lambda x: jnp.reshape(x, (self.sequence_len, self.fingering_dim))
        fingering_states = jax.vmap(fingering_reshaper)(fingering_states)
        fixed_state = jnp.expand_dims(fixed_state, 1)
        hidden_dim = self.config.hidden_size
        note_embeddings = jnp.concatenate([
            nn.Dense(features=hidden_dim)(goal_states[:, :1, :]),
            jax.vmap(self.note_conv)(goal_states[:, 1:, :]),
        ], axis=1)
        fingering_embeddings = jnp.concatenate([
            nn.Dense(features=hidden_dim)(fingering_states[:, :1, :]),
            jax.vmap(self.fingering_conv)(fingering_states[:, 1:, :]),
        ], axis=1)
        # note_embeddings = nn.Dense(features=hidden_dim, name="emb_goal")(note_embeddings)
        fixed_state_embedding = nn.Dense(features=hidden_dim, name="emb_fixed")(fixed_state)

        music_embeddings = note_embeddings + fingering_embeddings
        hidden_states = jnp.concatenate([music_embeddings, fixed_state_embedding], axis=1)

        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(hidden_states).shape[1]), hidden_states.shape[:-1])
        positional_embeddings = self.position_embeddings(position_ids.astype("i4"))

        hidden_states = hidden_states + positional_embeddings
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=not training)
        hidden_state = self.encoder(
            hidden_states,
            attention_mask=None,
            head_mask=None,
            deterministic=not training,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )[0]
        return hidden_state[:, 0, :]


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    sequence_len: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    obs_ts_dim: int = 89
    fingering_dim: int = 10

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        goal_states = x['goal']
        fingering_states = x['fingering']
        if len(x['reward'].shape) < len(x['goal'].shape):
            x['reward'] = jnp.expand_dims(x['reward'], -1)
        fixed_state = jnp.concatenate([
            v
            for k, v in x.items()
            if k not in ('goal', 'fingering')
        ], axis=-1)

        if len(fixed_state.shape) == 1:
            goal_states = jnp.expand_dims(goal_states, 0)
            fingering_states = jnp.expand_dims(fingering_states, 0)
            fixed_state = jnp.expand_dims(fixed_state, 0)
        bsz = goal_states.shape[0]

        goal_reshaper = lambda x: jnp.reshape(x, (self.sequence_len, self.obs_ts_dim))
        goal_states = jax.vmap(goal_reshaper)(goal_states)
        fingering_reshaper = lambda x: jnp.reshape(x, (self.sequence_len, self.fingering_dim))
        fingering_states = jax.vmap(fingering_reshaper)(fingering_states)
        note_embeddings = jnp.concatenate([
            goal_states[:, :1, :],
            jax.vmap(nn.Conv(goal_states.shape[-1], kernel_size=(10,), strides=(10,), name="note_conv"))(goal_states[:, 1:, :]),
        ], axis=1)
        fingering_embeddings = jnp.concatenate([
            fingering_states[:, :1, :],
            jax.vmap(nn.Conv(fingering_states.shape[-1], kernel_size=(10,), strides=(10,), name="fingering_conv"))(fingering_states[:, 1:, :]),
        ], axis=1)
        music_embeddings = jnp.concatenate([note_embeddings, fingering_embeddings], axis=-1).reshape((bsz, -1))
        x = jnp.concatenate([music_embeddings, fixed_state], axis=-1)
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activation(x)
        return x


class StateActionValue(nn.Module):
    base_cls: Type[nn.Module]

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        observations["actions"] = actions
        outputs = self.base_cls()(observations, *args, **kwargs)

        value = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1)


class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)


def subsample_ensemble(
    key: types.PRNGKey,
    params: types.Params,
    num_sample: Optional[int],
    num_qs: int,
) -> types.Params:
    if num_sample is not None:
        all_indx = jnp.arange(0, num_qs)
        indx = jax.random.choice(key, a=all_indx, shape=(num_sample,), replace=False)

        if "Ensemble_0" in params:
            ens_params = jax.tree_util.tree_map(
                lambda param: param[indx], params["Ensemble_0"]
            )
            params = params.copy(add_or_replace={"Ensemble_0": ens_params})
        else:
            params = jax.tree_util.tree_map(lambda param: param[indx], params)
    return params
