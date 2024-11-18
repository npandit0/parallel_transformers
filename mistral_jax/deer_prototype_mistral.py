"""
deer_prototype_mistral.py
this script sets up and runs a forward pass through the mistral 7b architecture
it's not that fast (we still have questions about whether jtu is parallelizing), and because we haven't found a nice solution for diagonal derivatives, we are very much limited by memory
However, as long as we use small T (sequence length) and batch size B, we can actually get this to run

TODOs:
* can we handle larger batch sizes and sequence lengths? what about on hardware?
* how is our accuracy? how can we get to bonafide exact recovery? and so clear the way to work on systems
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

    def __init__(self, dim, eps, dtype=jnp.bfloat16):
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

    def __init__(self, args, key, dtype=jnp.bfloat16):
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

        key, value = self.prefill(xk, xv)

        # 5. Output
        output = self.compute_scores_and_output(xq, key, value, mask, seqlen)
        return output


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

    # def __call__(self, x, cos_freq, sin_freq, positions, mask, cache_k, cache_v):
    def __call__(self, x, cos_freq, sin_freq, positions, mask):
        normed_x = jax.vmap(self.attention_norm)(x)
        r = self.attention(normed_x, cos_freq, sin_freq, positions, mask)
        h = x + r
        r = jax.vmap(self.feed_forward)(jax.vmap(self.ffn_norm)(h))
        out = h + r
        return out


class Transformer(eqx.Module):
    tok_embeddings: eqx.nn.Embedding
    layers: TransformerBlock
    norm: RMSNorm
    output: eqx.nn.Linear
    vocab_size: int
    n_layers: int
    sliding_window: int

    def __init__(self, args, key, dtype=jnp.bfloat16):
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
        t = jnp.full((seqlen, seqlen), dtype=jnp.bfloat16, fill_value=1)
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

    def __call__(self, x, cos_freq, sin_freq, positions, mask):
        """
        Edited to do prefilling instead of kv cache
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
            h = layer(h, cos_freq, sin_freq, positions, mask)  # h has shape (T,D)
            all_states.append(h)
            # print(f"at layer {i}, the shape of the feature is {h.shape}")

        # h = self.compute_norm(h)
        # h = self.compute_output(h).astype(jnp.float32)
        return jnp.array(all_states)

    def partial_layers(self, layers, cos_freq, sin_freq, positions, mask):
        """
        Ideally we could use jtu instead...
        """

        def partial_layer(layer):
            return partial(
                layer.__call__,
                cos_freq=cos_freq,
                sin_freq=sin_freq,
                positions=positions,
                mask=mask,
            )

        return [
            partial_layer(layer) for layer in layers
        ]  # really would prefer not to use list comprehension

    def parallel_call(self, x, cos_freq, sin_freq, positions, mask, num_iters=7):
        """
        Should give the same output as call, but using fixed point iterations
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
        # pdb.set_trace()
        partialed_layers = self.partial_layers(
            self.layers, cos_freq, sin_freq, positions, mask
        )
        states_guess = [
            jr.normal(jr.PRNGKey(i), (T, D))
            / jnp.sqrt(jnp.mean(jr.normal(jr.PRNGKey(i), (T, D)) ** 2))
            for i in range(num_layers)
        ]  # make sure to stay near rms norm equal to 1
        # states_guess = [jnp.zeros((T, D)) for _ in range(num_layers)] # never do this when using rms norm, grads will explode
        # calls out to deer
        all_states = deer(
            h0, partialed_layers, states_guess, num_iters
        )  # (batch_size, num_iters, num_layers, T, D)

        return jnp.array(all_states)
        # h = all_states[-1][-1]
        # pdb.set_trace()

        # h = self.compute_norm(h)
        # h = self.compute_output(h).astype(jnp.float32)
        # return h


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

    def step(states, args):
        """
        This step is a single deer iteration (will eventually be sequential scanned)
        Args:
          states: list of length num_layers of (T, D) shaped arrays
          args: None
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
        As = As.at[0].set(jnp.zeros((T, D, T, D)))
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
    for i in range(num_iters):
        print()
        print("-----------------")
        print(f"iteration {i}")
        print("-----------------")
        print()
        t1 = time.time()
        states_guess, iter_hist_add = step(states_guess, None)
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
            weight = torch_weights[path_pieces]
            weight = jnp.asarray(weight.float().numpy(), dtype=jnp.bfloat16)
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
    parser.add_argument("--num_iters", type=int, default=7, help="number of deer iterations. num_iters needs to be greater than 1 for plotting code to work")
    parser.add_argument(
        "--load_weights", action="store_true", help="Pre-load model weights"
    )
    parser.add_argument("--xavier", action="store_true", help="wandb login for xavier")
    args = parser.parse_args()

    if args.proto:
        args.load_weights = False
        args.num_iters = 2

    if args.xavier:
        wandb.init(project="parallel_transformer", entity="xavier_gonzalez")
    else:
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
        model_args, key=jax.random.PRNGKey(1), dtype=jnp.bfloat16
    )  # sets architecutre
    if(args.load_weights):
        state_dict = torch.load(
            "../model_files/consolidated.00.pth"
        )
        model = port_weights_from_torch(state_dict, model)  # fills with pretrained weights
    cos_freq, sin_freq = precompute_frequencies(model_args.head_dim, 128000)
    vmap_par = jax.vmap(
        partial(model.parallel_call, num_iters=args.num_iters),
        in_axes=(0, None, None, None, None),
    )  
    vmap_seq = jax.vmap(
        model,
        in_axes=(0, None, None, None, None),
    )
    fake_pos = jnp.array([0], dtype=jnp.int32)
    fake_inp = jnp.asarray([[1]], dtype=jnp.int32)
    fake_mask = None

    hist_seq = vmap_seq(
        fake_inp, cos_freq[fake_pos], sin_freq[fake_pos], fake_pos, fake_mask
    )
    hist_parr = vmap_par(
        fake_inp, cos_freq[fake_pos], sin_freq[fake_pos], fake_pos, fake_mask
    )

    results_dict = {
        "hist_seq": hist_seq,
        "hist_parr": hist_parr,
    }
    file_name = f"results_num_iters_{args.num_iters}"
    artifact = wandb.Artifact(file_name, type="dataset")
    save_to_pickle(results_dict, f"{file_name}.pkl")
    artifact.add_file(f"{file_name}.pkl")
    wandb.log_artifact(artifact)

    # make plots
    errors_per_iter_and_layer = jnp.mean(
        jnp.abs((hist_parr[0] - hist_seq).squeeze()), axis=-1
    )

    # error plot
    fig, ax = plt.subplots()
    ax.plot(errors_per_iter_and_layer[-1])

    # Add labels, log scale, and legend
    ax.set_xlabel("layers")
    ax.set_ylabel("mean error in activations")
    ax.set_yscale("log")
    ax.legend()
    # log to wandb
    wandb.log({"error_plot": wandb.Image(fig)})
    plt.close(fig)

    # heat map over all the iterations
    fig, ax = plt.subplots()
    # Plot the heatmap
    im = ax.imshow(errors_per_iter_and_layer, cmap="hot", norm=mcolors.LogNorm(), interpolation="nearest")
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax)
    wandb.log({"heatmap": wandb.Image(fig)})
    plt.close(fig)

    wandb.finish()
