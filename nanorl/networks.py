"""Neural network module."""

from typing import Any, Callable, Optional, Sequence, Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from trax.models.transformer import TransformerDecoder
from trax import layers as tl
from trax.layers import base

from nanorl import types

default_init = nn.initializers.xavier_uniform

# Shamelessly copied from Trax
# Defaults used across Transformer variants.
MODE = 'train'
D_MODEL = 512
D_FF = 2048
N_LAYERS = 6
N_HEADS = 8
MAX_SEQUENCE_LENGTH = 2048
DROPOUT_RATE = .1
DROPOUT_SHARED_AXES = None
FF_ACTIVATION_TYPE = tl.Relu

def _FeedForwardBlock(d_model,
                      d_ff,
                      dropout,
                      dropout_shared_axes,
                      mode,
                      activation):
    """Returns a list of layers that implements a feedforward block.

    Args:
        d_model: Last/innermost dimension of activation arrays at most points in
            the model, including the initial embedding output.
        d_ff: Last/innermost dimension of special (typically wider)
            :py:class:`Dense` layer in the feedforward part of each block.
        dropout: Stochastic rate (probability) for dropping an activation value
            when applying dropout within a block.
        dropout_shared_axes: Tensor axes on which to share a dropout mask.
            Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
            is a useful way to save memory and apply consistent masks to activation
            vectors at different sequence positions.
        mode: If ``'train'``, each block will include dropout; else, it will
            pass all values through unaltered.
        activation: Type of activation function at the end of each block; must
            be an activation-type subclass of :py:class:`Layer`.

    Returns:
        A list of layers that maps vectors to vectors.
    """
    def _Dropout():
        return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

    return [
      tl.Dense(d_ff),
      activation(),
      _Dropout(),
      tl.Dense(d_model),
    ]

def _EncoderBlock(d_model,
                  d_ff,
                  n_heads,
                  dropout,
                  dropout_shared_axes,
                  mode,
                  ff_activation):
    """Returns a list of layers that implements a Transformer encoder block.

    The input to the block is a pair (activations, mask) where the mask was
    created from the original source tokens to prevent attending to the padding
    part of the input. The block's outputs are the same type/shape as its inputs,
    so that multiple blocks can be chained together.

    Args:
        d_model: Last/innermost dimension of activation arrays at most points in
            the model, including the initial embedding output.
        d_ff: Last/innermost dimension of special (typically wider)
            :py:class:`Dense` layer in the feedforward part of each block.
        n_heads: Number of attention heads.
        dropout: Stochastic rate (probability) for dropping an activation value
            when applying dropout within encoder blocks. The same rate is also used
            for attention dropout in encoder blocks.
        dropout_shared_axes: Tensor axes on which to share a dropout mask.
            Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
            is a useful way to save memory and apply consistent masks to activation
            vectors at different sequence positions.
        mode: If ``'train'``, each block will include dropout; else, it will
            pass all values through unaltered.
        ff_activation: Type of activation function at the end of each block; must
            be an activation-type subclass of :py:class:`Layer`.

    Returns:
        A list of layers that act in series as a (repeatable) encoder block.
    """
    def _Attention():
        return tl.Attention(d_model, n_heads=n_heads, dropout=dropout, mode=mode)

    def _Dropout():
        return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

    def _FFBlock():
        return _FeedForwardBlock(d_model, d_ff, dropout, dropout_shared_axes, mode,
                                ff_activation)

    return [
        tl.Residual(
            tl.LayerNorm(),
            _Attention(),
            _Dropout(),
        ),
        tl.Residual(
            tl.LayerNorm(),
            _FFBlock(),
            _Dropout(),
        ),
    ]


class MusicalPositionalEncoding(base.Layer):
    def __init__(self, max_len=2048):
        super().__init__()
        self._max_len = max_len

    def init_weights_and_state(self, input_signature):
        super().init_weights_and_state(input_signature)
        d_feature = input_signature.shape[-1]
        pe = np.linspace(0.0, 1.0, self._max_len, dtype=np.float32)
        pe = np.broadcast_to(pe, (self._max_len, d_feature))
        w = jnp.array(pe)
        self.weights = w
    
    def forward(self, inputs):
        notes, indices = inputs
        assert notes.shape[1] == indices.shape[1]
        px = self.weights[indices, :]
        assert notes.shape == px.shape
        return px + notes
   


def MusicTransformerEncoder(vocab_size,
                       n_classes=10,
                       d_model=D_MODEL,
                       d_ff=D_FF,
                       n_layers=N_LAYERS,
                       n_heads=N_HEADS,
                       max_len=MAX_SEQUENCE_LENGTH,
                       dropout=DROPOUT_RATE,
                       dropout_shared_axes=DROPOUT_SHARED_AXES,
                       mode=MODE,
                       ff_activation=FF_ACTIVATION_TYPE):
  """Returns a Transformer encoder suitable for N-way classification.

  This model maps tokenized text to N-way (``n_classes``) activations:

    - input: Array representing a batch of text strings via token IDs plus
      padding markers; shape is (batch_size, sequence_length), where
      sequence_length <= ``max_len``. Array elements are integers in
      ``range(vocab_size)``, and 0 values mark padding positions.

    - output: Array representing a batch of raw (non-normalized) activations
      over ``n_classes`` categories; shape is (batch_size, ``n_classes``).

  Args:
    vocab_size: Input vocabulary size -- each element of the input array
        should be an integer in ``range(vocab_size)``. These integers typically
        represent token IDs from a vocabulary-based tokenizer.
    n_classes: Last/innermost dimension of output arrays, suitable for N-way
        classification.
    d_model: Last/innermost dimension of activation arrays at most points in
        the model, including the initial embedding output.
    d_ff: Last/innermost dimension of special (typically wider)
        :py:class:`Dense` layer in the feedforward part of each encoder block.
    n_layers: Number of encoder blocks. Each block includes attention, dropout,
        residual, layer-norm, feedforward (:py:class:`Dense`), and activation
        layers.
    n_heads: Number of attention heads.
    max_len: Maximum symbol length for positional encoding.
    dropout: Stochastic rate (probability) for dropping an activation value
        when applying dropout within encoder blocks. The same rate is also
        used for attention dropout in encoder blocks.
    dropout_shared_axes: Tensor axes on which to share a dropout mask.
        Sharing along batch and sequence axes (``dropout_shared_axes=(0,1)``)
        is a useful way to save memory and apply consistent masks to activation
        vectors at different sequence positions.
    mode: If ``'train'``, each encoder block will include dropout; else, it
        will pass all values through unaltered.
    ff_activation: Type of activation function at the end of each encoder
        block; must be an activation-type subclass of :py:class:`Layer`.

  Returns:
    A Transformer model that maps strings (conveyed by token IDs) to
    raw (non-normalized) activations over a range of output classes.
  """
  def _Dropout():
    return tl.Dropout(rate=dropout, shared_axes=dropout_shared_axes, mode=mode)

  def _EncBlock():
    return _EncoderBlock(d_model, d_ff, n_heads, dropout, dropout_shared_axes,
                         mode, ff_activation)

  return tl.Serial(
    #   tl.Branch([], tl.PaddingMask()),  # Creates masks from copy of the tokens.
      tl.Embedding(vocab_size, d_model),
      _Dropout(),
    #   tl.PositionalEncoding(max_len=max_len),
      [_EncBlock() for _ in range(n_layers)],
    #   tl.Select([0], n_in=2),  # Drops the masks.
      tl.LayerNorm(),
      tl.Mean(axis=1),
      tl.Dense(n_classes),
  )

class TransformerAnd(nn.Module):
    hidden_dim: int
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # TODO: extract the notes into the correct format and figure out how we're going to
        # push the state in
        x = TransformerDecoder(d_model=32, n_layers=2, mode='train' if training else None)(x)
        return x


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
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
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

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
