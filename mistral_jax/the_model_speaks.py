"""
the_model_speaks.py

Using the kv cache in conjunction with deer to get around memory bottlenek in computing the Jacobians
"""

import torch
import jax

import os
import gc
import time
import json
import pickle
import numpy as np

import torch
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from functools import partial
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float, Scalar
from typing import Optional, Tuple, List, NamedTuple

from sentencepiece import SentencePieceProcessor

import pdb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import argparse
import wandb
import pickle
import time

class Tokenizer:
    def __init__(self, model_path: str):
        self._model = SentencePieceProcessor(model_file=model_path)

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        return self._model.decode(t)


def precompute_frequencies(dim, max_pos, theta=10000.0):
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[: (dim // 2)] / dim)
    )
    t = jnp.arange(0, max_pos, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    return jnp.cos(freqs), jnp.sin(freqs)


@partial(jax.jit, static_argnums=(3,))
def calculate_rope(x, cos_freq, sin_freq, offset=0):
    # x shape  is [seqlen, num_heads, heads_dim]

    # Get the sequence length
    seqlen = x.shape[0]

    # Get the corresponding positional embeddings
    sin = sin_freq[offset : offset + seqlen, :]
    cos = cos_freq[offset : offset + seqlen, :]

    # Positional embeddings are 2D while our input is 3D
    # if `num_heads` dimension is present in the inputs.
    # We need to add another dimension to our positional embeddings
    sin = sin[:, jnp.newaxis, :]
    cos = cos[:, jnp.newaxis, :]

    # Get the even-odd positions from the inputs
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # Matmul with the rotation matrix
    # [cos_nθ, -sin_nθ] [x1]
    # [sin_nθ,  cos_nθ] [x2]
    # => [x1 * cos_nθ - x2 * sin_nθ, x1 * sin_nθ + x2 * cos_nθ]
    pos_embed = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    pos_embed = jax.lax.collapse(pos_embed, -2)
    return pos_embed.astype(x.dtype)


class RMSNorm(eqx.Module):
    """
    Make the root mean square of the output to be 1.
    """

    eps: float
    weight: Float[Array, "*shape"]

    def __init__(self, dim, eps, dtype=jnp.float32):
        dtype = default_floating_dtype if dtype is None else dtype
        self.eps = eps
        self.weight = jnp.ones(shape=dim, dtype=dtype)

    def rmsnorm(self, x):
        return jnp.sqrt(jnp.mean(x**2, keepdims=True) + self.eps)

    def _norm(self, x):
        return x * jax.lax.rsqrt(jnp.mean(x**2, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(jnp.float32)).astype(x.dtype)
        return output * self.weight


class FeedForward(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear

    def __init__(self, args, key, dtype=jnp.float32):
        dtype = default_floating_dtype if dtype is None else dtype
        key1, key2, key3 = jax.random.split(key, 3)

        self.w1 = eqx.nn.Linear(
            args.dim, args.hidden_dim, use_bias=False, key=key1, dtype=dtype
        )
        self.w2 = eqx.nn.Linear(
            args.hidden_dim, args.dim, use_bias=False, key=key2, dtype=dtype
        )
        self.w3 = eqx.nn.Linear(
            args.dim, args.hidden_dim, use_bias=False, key=key3, dtype=dtype
        )

    def __call__(self, x):
        h = jax.nn.silu(self.w1(x).astype(jnp.float32)).astype(x.dtype)
        return self.w2(h * self.w3(x))


class Attention(eqx.Module):
    dim: int
    n_heads: int
    head_dim: int
    n_kv_heads: int
    kv_repeats: int
    sliding_window: int
    scale: float
    wq: eqx.nn.Linear
    wk: eqx.nn.Linear
    wv: eqx.nn.Linear
    wo: eqx.nn.Linear

    def __init__(self, args, key, dtype=jnp.bfloat16):
        dtype = default_floating_dtype if dtype is None else dtype
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.n_kv_heads = args.n_kv_heads
        self.dim = args.dim
        self.kv_repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = args.sliding_window

        self.scale = args.head_dim**-0.5

        self.wq = eqx.nn.Linear(
            args.dim,
            args.n_heads * args.head_dim,
            use_bias=False,
            key=key1,
            dtype=dtype,
        )
        self.wk = eqx.nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            use_bias=False,
            key=key2,
            dtype=dtype,
        )
        self.wv = eqx.nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            use_bias=False,
            key=key3,
            dtype=dtype,
        )
        self.wo = eqx.nn.Linear(
            args.n_heads * args.head_dim,
            args.dim,
            use_bias=False,
            key=key4,
            dtype=dtype,
        )

    @partial(jax.jit, static_argnums=(2, 3))
    def get_cache_slice(self, x, pos, kv_repeats):
        x_slice = x.at[:pos, :, :].get()
        x_slice = jnp.repeat(x_slice, kv_repeats, axis=1)
        return x_slice

    @eqx.filter_jit
    def compute_qkv(self, x):
        seqlen, _ = x.shape

        xq = jax.vmap(self.wq)(x)
        xk = jax.vmap(self.wk)(x)
        xv = jax.vmap(self.wv)(x)

        xq = jnp.reshape(xq, (seqlen, self.n_heads, self.head_dim))
        xk = jnp.reshape(xk, (seqlen, self.n_kv_heads, self.head_dim))
        xv = jnp.reshape(xv, (seqlen, self.n_kv_heads, self.head_dim))
        return xq, xk, xv

    @jax.jit
    def update_cache_values(self, xk, xv, cache_k, cache_v, positions):
        cache_k = cache_k.at[positions, ...].set(xk[positions, ...])
        cache_v = cache_v.at[positions, ...].set(xv[positions, ...])
        return cache_k, cache_v

    @eqx.filter_jit
    def prefill(self, xk, xv):
        key = jnp.repeat(xk, self.kv_repeats, axis=1)
        value = jnp.repeat(xv, self.kv_repeats, axis=1)
        return key, value

    @eqx.filter_jit
    def compute_scores_and_output(self, xq, key, value, mask, seqlen):
        query = jnp.transpose(xq, (1, 0, 2))
        key = jnp.transpose(key, (1, 0, 2))
        value = jnp.transpose(value, (1, 0, 2))

        # # # scores : [n_heads, seqlen | 1, seqlen]
        scores = jnp.matmul(query, jnp.transpose(key, (0, 2, 1))) * self.scale

        if mask is not None:
            # Mask will of shape [seqlen, seqlen] but our scores
            # have shape [num_heads, seqlen, seqlen], hence we need
            # to introduce another dimension in the mask
            mask = mask[jnp.newaxis, ...]
            scores = scores + mask

        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
        output = jnp.matmul(scores, value)
        output = jnp.reshape(jnp.transpose(output, (1, 0, 2)), (seqlen, -1))
        output = jax.vmap(self.wo)(output)
        return output

    def __call__(
        self, x, cos_freq, sin_freq, positions, mask=None, cache_k=None, cache_v=None
    ):
        # x shape: [seqlen, embed_dim]
        seqlen, _ = x.shape
        # 1. Calculate qkv
        xq, xk, xv = self.compute_qkv(x)

        # 2. Calculate RoPE
        xq = calculate_rope(xq, cos_freq, sin_freq, 0)
        xk = calculate_rope(xk, cos_freq, sin_freq, 0)

        # 3. Update cache
        # pdb.set_trace()
        cache_k, cache_v = self.update_cache_values(xk, xv, cache_k, cache_v, positions)

        # 4. Generation
        if positions.shape[0] > 1:
            # prefill
            key, value = self.prefill(xk, xv)
        else:
            # single-token generation
            cur_pos = positions[-1].item() + 1
            key = self.get_cache_slice(cache_k, cur_pos, self.kv_repeats)
            value = self.get_cache_slice(cache_v, cur_pos, self.kv_repeats)

        # 5. Output
        output = self.compute_scores_and_output(xq, key, value, mask, seqlen)
        return output, cache_k, cache_v


class TransformerBlock(eqx.Module):
    dim: int
    n_heads: int
    attention: Attention
    attention_norm: RMSNorm
    feed_forward: FeedForward
    ffn_norm: RMSNorm

    def __init__(self, args, key, dtype=jnp.bfloat16):
        key1, key2 = jax.random.split(key, 2)
        self.n_heads = args.n_heads
        self.dim = args.dim

        self.attention = Attention(args, key=key1, dtype=dtype)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)

        self.feed_forward = FeedForward(args, key=key2, dtype=dtype)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)

    def __call__(self, x, cos_freq, sin_freq, positions, mask, cache_k, cache_v):
        normed_x = jax.vmap(self.attention_norm)(x)
        r, cache_k, cache_v = self.attention(
            normed_x, cos_freq, sin_freq, positions, mask, cache_k, cache_v
        )
        h = x + r
        r = jax.vmap(self.feed_forward)(jax.vmap(self.ffn_norm)(h))
        out = h + r
        return out, cache_k, cache_v


class Transformer(eqx.Module):
    tok_embeddings: eqx.nn.Embedding
    layers: TransformerBlock
    norm: RMSNorm
    output: eqx.nn.Linear
    vocab_size: int
    n_layers: int
    sliding_window: int

    def __init__(self, args, key, dtype=jnp.float32):
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.sliding_window = args.sliding_window
        keys = jax.random.split(key, args.n_layers + 2)
        embed_key, linear_key, tf_layers_keys = keys[0], keys[1], keys[2:]

        self.tok_embeddings = eqx.nn.Embedding(
            args.vocab_size, args.dim, key=embed_key, dtype=dtype
        )
        self.norm = RMSNorm(dim=args.dim, eps=args.norm_eps, dtype=dtype)
        self.output = eqx.nn.Linear(
            args.dim, args.vocab_size, use_bias=False, key=linear_key, dtype=dtype
        )
        self.layers = [
            TransformerBlock(args, key=tf_layers_keys[i], dtype=dtype)
            for i in range(args.n_layers)
        ]

    @eqx.filter_jit
    def compute_embeddings(self, x):
        return jax.vmap(self.tok_embeddings)(x)

    @eqx.filter_jit
    def compute_mask(self, seqlen):
        t = jnp.full((seqlen, seqlen), dtype=jnp.float32, fill_value=1)
        mask = jnp.tril(t, k=0)
        # make the mask banded to account for sliding window
        mask = jnp.triu(mask, k=-self.sliding_window)
        mask = jnp.log(mask)
        return mask

    @eqx.filter_jit
    def compute_norm(self, x):
        return jax.vmap(self.norm)(x)

    @eqx.filter_jit
    def compute_output(self, x):
        return jax.vmap(self.output)(x)

    @partial(jax.jit, static_argnums=(1,))
    def update_cache_values(
        self, idx, cache_k, cache_v, cache_k_updates, cache_v_updates
    ):
        cache_k = cache_k.at[idx, :, :, :].set(cache_k_updates)
        cache_v = cache_v.at[idx, :, :, :].set(cache_v_updates)
        return cache_k, cache_v

    def __call__(self, x, cos_freq, sin_freq, positions, mask, cache_k, cache_v):
        """
        The return signature is logits, cache_k, cache_v, list of intermediate activations (xs)
        """
        # x is of shape (seqlen, )
        h = self.compute_embeddings(x)

        if x.shape[-1] > 1:
            seqlen = x.shape[-1]
            mask = self.compute_mask(seqlen)
        else:
            mask = None

        all_states = []
        for i, layer in enumerate(self.layers):
            # h has shape (len(positions), dim)
            # cache_ki has shape (sliding_window_len, head_dim, n_kv_heads)
            h, cache_ki, cache_vi = layer(
                h, cos_freq, sin_freq, positions, mask, cache_k[i, ...], cache_v[i, ...]
            )  # h has shape (T,D)
            cache_k, cache_v = self.update_cache_values(
                i, cache_k, cache_v, cache_ki, cache_vi
            )
            all_states.append(h)
            # print(f"at layer {i}, the shape of the feature is {h.shape}")

        h = self.compute_norm(h)
        h = self.compute_output(h).astype(jnp.float32)
        return h, cache_k, cache_v, jnp.array(all_states)

    def partial_layers(self, layers, cos_freq, sin_freq, positions, mask, cache_k, cache_v):
        """
        cache_k: shape (L, sliding_window, num_heads, head_dim)
        Ideally we could use jtu instead...
        """
        x=1

        def partial_layer(layer, index):
            return partial(
                layer.__call__,
                cos_freq=cos_freq,
                sin_freq=sin_freq,
                positions=positions,
                mask=mask,
                cache_k=cache_k[index], 
                cache_v=cache_v[index]
            )

        return [
            lambda state : partial_layer(layer,i)(state)[0] for i,layer in enumerate(layers)
        ]  # really would prefer not to use list comprehension

    def parallel_call(self, x, cos_freq, sin_freq, positions, mask, cache_k, cache_v, num_iters=7):
        """
        Should give the same output as call, but using fixed point iterations
        x is a tensor of token indices (B,T)
        """
        h0 = self.compute_embeddings(x)
        T, D = h0.shape

        if x.shape[-1] > 1:
            seqlen = x.shape[-1]
            mask = self.compute_mask(seqlen)
        else:
            mask = None

        # parallel logic
        num_layers = len(self.layers)
        partialed_layers = self.partial_layers(
            self.layers, cos_freq, sin_freq, positions, mask, cache_k, cache_v
        )
        states_guess = [
            jr.normal(jr.PRNGKey(i), (T, D), dtype=jnp.float32)
            / jnp.sqrt(jnp.mean(jr.normal(jr.PRNGKey(i), (T, D), dtype=jnp.float32) ** 2))
            for i in range(num_layers)
        ]  # make sure to stay near rms norm equal to 1
        # calls out to deer
        all_states = deer(
            h0, partialed_layers, states_guess, num_iters
        )  # (num_iters, num_layers, T, D)

        # loop through all_states to the kv cache
        for i, layer in enumerate(self.layers):
            _, cache_ki, cache_vi = layer(
                all_states[-1, i], cos_freq, sin_freq, positions, mask, cache_k[i, ...], cache_v[i, ...]
            )
            # pdb.set_trace()
            cache_k, cache_v = self.update_cache_values(
                i, cache_k, cache_v, cache_ki, cache_vi
            )  
        h = all_states[-1][-1]
        h = self.compute_norm(h)
        logits = self.compute_output(h).astype(jnp.float32)
        return logits, cache_k, cache_v, jnp.array(all_states)


def deer(x, layers, states_guess, num_iters, k=1):
    """
    runs deer (fiddly logic in the rearrange)

    Args:
      x: (T, d) initial inputs to transformer stack
      layers: list of TransformerLayer objects (the functions that propagate information over the stack)
      states_guess: list of length num_layers of (T, D) shaped arrays; this is the initial guess for the states. don't make them all zero!
      num_iters: number of iterations to run for
      k: damping factor
    """
    T, D = x.shape
    num_layers = len(layers)

    @jax.vmap
    def binary_op(q_i, q_j):
        """Binary operator for parallel scan of linear recurrence. Assumes a full Jacobian matrix A
        Args:
            q_i: tuple containing J_i and b_i at position i       (P,P), (P,)
            q_j: tuple containing J_j and b_j at position j       (P,P), (P,)
        Returns:
            new element ( A_out, Bu_out )
        """
        A_i, b_i = q_i
        A_j, b_j = q_j
        return A_j @ A_i, A_j @ b_i + b_j

    def rms_normalize(arr, eps):
        """
        Apply rms normalization to an array
        """
        return arr * jax.lax.rsqrt(jnp.mean(arr ** 2) + eps)

    def step(states, iter_num):
        """
        This step is a single deer iteration (will eventually be sequential scanned)
        Args:
          states: list of length num_layers of (T, D) shaped arrays
          args: this needs to be a range from 1 to num_iters
        """
        states = [x] + states[:-1]  # length num_layers
        print(f"states shape is {states[0].shape}")
        fs = jnp.array(
            jtu.tree_map(lambda x, f: f(x), states, layers)
        )  # (num_layers, T,D) arrays, note that we keep states as a list so we can use jtu.tree_map
        print(f"fs shape is {fs.shape}")
        As = jnp.array(
            jtu.tree_map(
                lambda x, f: jax.jacrev(f)(x), states, layers
            )  # this line seems to be the bottleneck, but that's odd bc we'd expect someone to take this grads during backprop
        )  # (num_layers, T, D, T, D) tensors
        print(f"As shape is {As.shape}")
        As = As.at[0:iter_num].set(jnp.zeros((iter_num, T, D, T, D))) # hard code reset
        # pdb.set_trace()
        # need to make the first A equal to zero
        states = jnp.array(states)  # (num_layers, T,D)
        # do some rearranging
        print("starting the rearranges")
        flattened_states = jnp.reshape(states, (num_layers, T * D))  # (num_layers, T*D)
        flattened_As = jnp.reshape(
            As, (num_layers, T * D, T * D)
        )  # (num_layers, T*D, T*D)
        # somehow, we aren't even getting here
        # print("we are about to start computing eigenvalues")
        # for A in flattened_As:
        #     print(jnp.linalg.eigvals(A))
        # pdb.set_trace()
        flattened_fs = jnp.reshape(fs, (num_layers, T * D))  # (num_layers, T*D)
        bs = flattened_fs - jnp.einsum(
            "tij,tj->ti", flattened_As, flattened_states
        )  # (num_layers, T*D)

        # finally ready to evaluate linearized dynamics (in parallel)
        print("we are about to start the associative scan")
        _, new_states = jax.lax.associative_scan(
            binary_op, (flattened_As, bs)
        )  # parallel operation
        # new_states = jnp.nan_to_num(new_states)  # zero out nans, (num_layers, T*D)
        new_states = jnp.reshape(new_states, (num_layers, T, D))  # (num_layers, T, D)
        return list(new_states), new_states

    print("starting deer outer loop")
    # TODO: use a scan instead of a for loop
    # states_guess, iter_hist = jax.lax.scan(
    #     step, states_guess, None, length=num_iters
    # )  # state_iters will show all the intermediate traces

    iter_hist = []
    # iter_nums = jnp.arange(1, num_iters+1)
    for i in range(num_iters):
        print()
        print("-----------------")
        print(f"iteration {i}")
        print("-----------------")
        print()
        t1 = time.time()
        # states_guess, iter_hist_add = step(states_guess, 1)
        states_guess, iter_hist_add = step(states_guess, i+1)
        t2 = time.time()
        wandb.log({"time_per_iter": t2 - t1})
        iter_hist.append(iter_hist_add)

    # return states_guess
    return jnp.array(iter_hist)  # (num_iters, num_layers, T, D)


class ModelArgs(NamedTuple):
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    vocab_size: int
    sliding_window: int
    norm_eps: float
    max_batch_size: int = 1


def port_weights_from_torch(torch_weights, eqx_model):
    def load_weights(path, leaf):
        path_pieces = []
        for path_elem in path:
            if isinstance(path_elem, jax.tree_util.GetAttrKey):
                path_pieces.append(path_elem.name)
            elif isinstance(path_elem, jax.tree_util.SequenceKey):
                path_pieces.append(str(path_elem.idx))
            else:
                raise ValueError(f"Unsupported path type {type(path_elem)}")

        path_pieces = ".".join(path_pieces)

        if "weight" in path_pieces:
            # print(path_pieces)
            weight = torch_weights[path_pieces]
            weight = jnp.asarray(weight.float().numpy(), dtype=jnp.float32)
            assert weight.shape == leaf.shape
            assert weight.dtype == leaf.dtype
            return weight
        else:
            print(f"Weights not ported for: {path_pieces}")
            return leaf

    return jax.tree_util.tree_map_with_path(load_weights, eqx_model)


def save_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def generate(model, tokenizer, cache_k, cache_v, head_dim, max_tokens=36, parallel=True, num_iters=2):
    cos_freq, sin_freq = precompute_frequencies(head_dim, 128000)
    parallel_model = jax.vmap(
            model.parallel_call, in_axes=(0, None, None, None, None, 0, 0, None)
        ) 
    sequential_model = jax.vmap(
            model, in_axes=(0, None, None, None, None, 0, 0)
        )
    # 1. Encode the prompts
    prompts = ["This is another test"]
    encoded_prompts = [
        tokenizer.encode(prompt) for prompt in prompts
    ]  # a list of lists
    # pdb.set_trace()
    # print(encoded_prompts)
    # encoded_prompts[0] = encoded_prompts[0][1:]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    # 2. Using numpy to generate the desired input. Will replace it with something
    # better later on
    input_tokens = np.full(
        (len(prompts), max_prompt_len), tokenizer.pad_id, dtype=np.int32
    )
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = jnp.array((encoded))
    input_mask = input_tokens != tokenizer.pad_id
    cur_pos = min_prompt_len

    # 3. pre-fill
    positions = jnp.arange(0, min_prompt_len)
    start = time.time()
    logits, cache_k, cache_v, _ = sequential_model(
        jnp.asarray(input_tokens[:, :min_prompt_len]),
        cos_freq[positions],
        sin_freq[positions],
        positions,
        None,
        cache_k,
        cache_v,
    )
    print(f"Time taken to prefill: {time.time()- start :.2f} seconds")
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    next_token = jnp.argmax(logprobs[:, -1, :], axis=-1)

    # 4. Generation
    generated = [next_token[0].item()]
    print("Generating...")
    all_logits = []
    final_layers = []
    start = time.time()
    for _ in range(max_tokens):
        cur_pos += 1
        pos = jnp.array([cur_pos])
        if parallel:
            logits, cache_k, cache_v, all_layers = parallel_model(
                jnp.asarray(next_token[:, None]),
                cos_freq[pos],
                sin_freq[pos],
                pos,
                None,
                cache_k,
                cache_v,
                num_iters
            )
        else:
            logits, cache_k, cache_v, all_layers = sequential_model(
                jnp.asarray(next_token[:, None]),
                cos_freq[pos],
                sin_freq[pos],
                pos,
                None,
                cache_k,
                cache_v,
            )
        all_logits.append(logits)
        # pdb.set_trace()
        final_layers.append(all_layers[0]) # all_layers[0] is an array with shape (L, T, D)
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        next_token = jnp.argmax(logprobs[:, -1, :], axis=-1)
        generated.append(next_token[0].item())

    end = time.time()
    res = prompts[0] + " " + "".join(tokenizer.decode(generated))
    print(res, "\n")
    print(f"Time taken to generate {max_tokens} tokens: {end- start :.2f} seconds")
    return res, generated, all_logits, final_layers


if __name__ == "__main__":

    import os
    import logging

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Check if we're forcing CPU usage
    force_cpu = os.environ.get('JAX_PLATFORM_NAME') == 'cpu' or os.environ.get('JAX_PLATFORMS') == 'cpu'

    if force_cpu:
        logging.info("Forcing CPU usage based on environment variables")
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import jax
    import jax.numpy as jnp

    # Log the devices JAX can see
    logging.info(f"JAX devices: {jax.devices()}")

    if not force_cpu:
        try:
            # Try to create a GPU array
            jax.device_put(jnp.zeros(1), jax.devices("gpu")[0])
            logging.info("Successfully initialized GPU")
        except RuntimeError as e:
            logging.warning(f"Failed to initialize GPU: {e}")
            logging.info("Falling back to CPU")
            jax.config.update('jax_platform_name', 'cpu')

    # Log the final platform being used
    logging.info(f"JAX is using platform: {jax.default_backend()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--proto", action="store_true", help="use prototype parameters")
    parser.add_argument("--num_iters", type=int, default=32, help="number of deer iterations. num_iters needs to be greater than 1 for plotting code to work")
    parser.add_argument("--num_tokens", type=int, default=2, help="number of tokens to generate")
    parser.add_argument(
        "--load_weights", action="store_true", help="Pre-load model weights"
    )
    # parser.add_argument(
    #     "--parallel", action="store_true", help="generate in parallel"
    # )
    args = parser.parse_args()

    if args.proto:
        args.load_weights = False
        args.num_iters = 8

    wandb.init(project="parallel_transformer")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.load_weights:
        param_file = "../model_files/params.json"
    else:
        param_file = "../model_files/proto_params.json"

    with open(param_file, "r") as f:
        model_args = ModelArgs(**json.loads(f.read()))

    model = Transformer(
        model_args, key=jax.random.PRNGKey(1), dtype=jnp.float32
    )  # sets architecture
    # pdb.set_trace()
    if args.load_weights:
        state_dict = torch.load("../model_files/consolidated.00.pth")
        model = port_weights_from_torch(state_dict, model)

    cache_k = jnp.zeros((model_args.max_batch_size, model_args.n_layers, model_args.sliding_window, model_args.n_kv_heads, model_args.head_dim), dtype=jnp.float32)
    cache_v = jnp.zeros((model_args.max_batch_size, model_args.n_layers, model_args.sliding_window, model_args.n_kv_heads, model_args.head_dim), dtype=jnp.float32)

    tokenizer = Tokenizer("../model_files/tokenizer.model")

    #seq generation
    res_seq, gen_seq, seq_logits, seq_finals = generate(
        model,
        tokenizer,
        cache_k,
        cache_v,
        model_args.head_dim,
        max_tokens=args.num_tokens,
        parallel=False,
    )
    print(f"the output of sequential is : {res_seq}")
    print(f"the generated tokens from sequential is : {gen_seq}")
    #pdb.set_trace()

    # parr generation
    res_parr, gen_parr, parr_logits, parr_finals = generate(model, tokenizer, cache_k, cache_v, model_args.head_dim, max_tokens=args.num_tokens, parallel=True, num_iters=args.num_iters)
    print(f"the output of parallel is : {res_parr}")
    print(f"the genrated tokens from parallel is : {gen_parr}")
    # pdb.set_trace()

    for i in range(len(seq_logits)):
        plt.plot(seq_logits[i][0,0] - parr_logits[i][0,0])
        plt.xlabel("token id")    
        plt.ylabel("logit difference (seq - parr)")
        plt.title(f"Logit difference between sequential and parallel at token {i}")
        plt.show()
        plt.savefig(f"logit_diff_{i}.png")

    for i in range(len(seq_finals)):
        plt.plot(jnp.mean(seq_finals[i][-1,0] - parr_finals[i][-1,0]))
        plt.xlabel("token id")    
        plt.ylabel("Mean difference in final layer (seq - parr)")
        plt.title(f"Mean difference in final layer between sequential and parallel at token {i}")
        plt.show()
        plt.savefig(f"layer_diff_{i}.png")



    pdb.set_trace()