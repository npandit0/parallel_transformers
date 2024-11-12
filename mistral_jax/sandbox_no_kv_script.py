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
import jax.tree_util as jtu

from functools import partial
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float, Scalar
from typing import Optional, Tuple, List, NamedTuple

from sentencepiece import SentencePieceProcessor

import pdb

# Set device to CPU for torch
device  = torch.device("cpu")

# Load the model dict, and check if any GPU is used
# state_dict = torch.load("mistral-7B-v0.1/consolidated.00.pth")
state_dict = torch.load(
    "/Users/xaviergonzalez/Desktop/xavier_folders/stanford/cs229s/mistral_jax/model_files/consolidated.00.pth"
)

# Tokenizer class
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
    eps: float
    weight: Float[Array, "*shape"]

    def __init__(self, dim, eps, dtype=jnp.bfloat16):
        dtype = default_floating_dtype if dtype is None else dtype
        self.eps = eps
        self.weight = jnp.ones(shape=dim, dtype=dtype)

    def _norm(self, x):
        return x * jax.lax.rsqrt(jnp.mean(x **2 , keepdims=True) + self.eps)

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

        self.w1 = eqx.nn.Linear(args.dim, args.hidden_dim, use_bias=False, key=key1, dtype=dtype)
        self.w2 = eqx.nn.Linear(args.hidden_dim, args.dim, use_bias=False, key=key2, dtype=dtype)
        self.w3 = eqx.nn.Linear(args.dim, args.hidden_dim, use_bias=False, key=key3, dtype=dtype)

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

        self.wq = eqx.nn.Linear(args.dim, args.n_heads * args.head_dim, use_bias=False, key=key1, dtype=dtype)
        self.wk = eqx.nn.Linear(args.dim, args.n_kv_heads * args.head_dim, use_bias=False, key=key2, dtype=dtype)
        self.wv = eqx.nn.Linear(args.dim, args.n_kv_heads * args.head_dim, use_bias=False, key=key3, dtype=dtype)
        self.wo = eqx.nn.Linear(args.n_heads * args.head_dim, args.dim, use_bias=False, key=key4, dtype=dtype)

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

    def __call__(self,  x, cos_freq, sin_freq, positions, mask=None, cache_k=None, cache_v=None):
        # x shape: [seqlen, embed_dim]
        seqlen, _ = x.shape
        # 1. Calculate qkv
        xq, xk, xv = self.compute_qkv(x)

        # 2. Calculate RoPE
        xq = calculate_rope(xq, cos_freq, sin_freq, 0)
        xk = calculate_rope(xk, cos_freq, sin_freq, 0)

        key, value = self.prefill(xk, xv)

        # # 3. Update cache
        # cache_k, cache_v = self.update_cache_values(xk, xv, cache_k, cache_v, positions)

        # # 4. Generation
        # if positions.shape[0] > 1:
        #     # prefill
        #     key, value = self.prefill(xk, xv)
        # else:
        #     # single-token generation
        #     cur_pos = positions[-1].item() + 1
        #     key = self.get_cache_slice(cache_k, cur_pos, self.kv_repeats)
        #     value = self.get_cache_slice(cache_v, cur_pos, self.kv_repeats)

        # 5. Output
        output = self.compute_scores_and_output(xq, key, value, mask, seqlen)
        # return output, cache_k, cache_v
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
        # r, cache_k, cache_v = self.attention(normed_x, cos_freq, sin_freq, positions, mask, cache_k, cache_v)
        r = self.attention(
            normed_x, cos_freq, sin_freq, positions, mask
        )
        h = x + r
        r = jax.vmap(self.feed_forward)(jax.vmap(self.ffn_norm)(h))
        out = h + r
        return out
        # return out, cache_k, cache_v

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

        self.tok_embeddings = eqx.nn.Embedding(args.vocab_size, args.dim, key=embed_key, dtype=dtype)
        self.norm = RMSNorm(dim=args.dim, eps=args.norm_eps, dtype=dtype)
        self.output = eqx.nn.Linear(args.dim, args.vocab_size, use_bias=False, key=linear_key, dtype=dtype)
        self.layers = [TransformerBlock(args, key=tf_layers_keys[i], dtype=dtype) for i in range(args.n_layers)] 

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
    def update_cache_values(self, idx, cache_k, cache_v, cache_k_updates, cache_v_updates):
        cache_k = cache_k.at[idx, :, :, :].set(cache_k_updates)
        cache_v = cache_v.at[idx, :, :, :].set(cache_v_updates)
        return cache_k, cache_v

    # def __call__(self, x, cos_freq, sin_freq, positions, mask, cache_k, cache_v):
    #     # x is of shape (seqlen, )
    #     h = self.compute_embeddings(x)

    #     if x.shape[-1] > 1:
    #         seqlen = x.shape[-1]
    #         mask = self.compute_mask(seqlen)
    #     else:
    #         mask = None

    #     # the for loop!!!

    #     for i, layer in enumerate(self.layers):
    #         #pdb.set_trace()
    #         # h has shape (len(positions), dim)
    #         # cache_ki has shape (sliding_window_len, head_dim, n_kv_heads)
    #         h, cache_ki, cache_vi = layer(h, cos_freq, sin_freq, positions, mask, cache_k[i, ...], cache_v[i, ...]) # I think we could get away with creating blank entries for h, cache_ki, and cache_vi
    #         # pdb.set_trace()
    #         cache_k, cache_v = self.update_cache_values(i, cache_k, cache_v, cache_ki, cache_vi) # I think all this line is doing is plugging in cache_ki and cache_vi in the appropriate palce

    #     h = self.compute_norm(h)
    #     h = self.compute_output(h).astype(jnp.float32)
    #     return h, cache_k, cache_v

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

        for i, layer in enumerate(self.layers):
            # h has shape (len(positions), dim)
            # cache_ki has shape (sliding_window_len, head_dim, n_kv_heads)
            h = layer(h, cos_freq, sin_freq, positions, mask) # h has shape (T,D)
            # print(f"at layer {i}, the shape of the feature is {h.shape}")

        h = self.compute_norm(h)
        h = self.compute_output(h).astype(jnp.float32)
        return h
    
    def partial_layers(self, layers, cos_freq, sin_freq, positions, mask):
        """
        Ideally we could use jtu instead...
        """
        def partial_layer(layer):
            return partial(layer.__call__, cos_freq=cos_freq, sin_freq=sin_freq, positions=positions, mask=mask)
        
        return [partial_layer(layer) for layer in layers] # really would prefer not to use list comprehension

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
        partialed_layers = self.partial_layers(self.layers, cos_freq, sin_freq, positions, mask)
        states_guess = [jnp.zeros((T, D)) for _ in range(num_layers)] # we can probably play around with smarter initialization strategies, too
        all_states = deer(h0, partialed_layers, states_guess, num_iters) # (num_iters, num_layers, T, D)
        h = all_states[-1, -1]
        # pdb.set_trace()

        h = self.compute_norm(h)
        h = self.compute_output(h).astype(jnp.float32)
        return h


def deer(x, layers, states_guess, num_iters):
    """
    runs deer (fiddly logic in the rearrange)

    Args:
      x: (T, d) initial inputs to transformer stack
      layers: list of TransformerLayer objects (the functions that propagate information over the stack)
      states_guess: list of length num_layers of (T, D) shaped arrays
      num_iters: number of iterations to run for
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

    def step(states, args):
        """
        This step is a single deer iteration (will eventually be sequential scanned)
        Args:
          states: list of length num_layers of (T, D) shaped arrays
          args: None
        """
        states = [x] + states[:-1]  # length num_layers
        fs = jnp.array(
            jtu.tree_map(lambda x, f: f(x), states, layers)
        )  # (num_layers, T,D) arrays, note that we keep states as a list so we can use jtu.tree_map
        As = jnp.array(
            jtu.tree_map(lambda x, f: jax.jacrev(f)(x), states, layers)
        )  # (num_layers, T, D, T, D) tensors
        As = As.at[0].set(jnp.zeros((T, D, T, D)))
        # pdb.set_trace()
        # need to make the first A equal to zero
        states = jnp.array(states)  # (num_layers, T,D)
        # do some rearranging
        flattened_states = jnp.reshape(states, (num_layers, T * D))  # (num_layers, T*D)
        flattened_As = jnp.reshape(
            As, (num_layers, T * D, T * D)
        )  # (num_layers, T*D, T*D)
        print(f"{jnp.eigvals(flattened_As[0])}")
        # pdb.set_trace()
        flattened_fs = jnp.reshape(fs, (num_layers, T * D))  # (num_layers, T*D)
        bs = flattened_fs - jnp.einsum(
            "tij,tj->ti", flattened_As, flattened_states
        )  # (num_layers, T*D)

        # finally ready to evaluate linearized dynamics (in parallel)
        _, new_states = jax.lax.associative_scan(
            binary_op, (flattened_As, bs)
        )  # parallel operation
        # new_states = jnp.nan_to_num(new_states)  # zero out nans, (num_layers, T*D)
        new_states = jnp.reshape(new_states, (num_layers, T, D))  # (num_layers, T, D)
        return list(new_states), new_states

    print("starting deer outer loop")
    # _, iter_hist = jax.lax.scan(
    #     step, states_guess, None, length=num_iters
    # )  # state_iters will show all the intermediate traces

    for i in range(num_iters):
        print()
        print("-----------------")
        print(f"iteration {i}")
        print("-----------------")
        print()
        states_guess, _ = step(states_guess, None)

    return states_guess
    # return iter_hist

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




def main():
    with open(
    "/Users/xaviergonzalez/Desktop/xavier_folders/stanford/cs229s/mistral_jax/model_files/proto_params.json",
    "r",
) as f:
    # with open('./mistral-7B-v0.1/params.json', 'r') as f:
        args = ModelArgs(**json.loads(f.read()))
    model = Transformer(args, key=jax.random.PRNGKey(1), dtype=jnp.bfloat16) # sets architecutre
    cos_freq, sin_freq = precompute_frequencies(args.head_dim, 128000)
    vmapped = jax.vmap(partial(model.parallel_call, num_iters=15), in_axes=(0, None, None, None, None)) # vmapped is the name of the model
    fake_pos = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    fake_inp = jnp.asarray([[1,  832,  349,  265, 1369]], dtype=jnp.int32)
    fake_mask = None

    # warmup
    logits = vmapped(fake_inp, cos_freq[fake_pos], sin_freq[fake_pos], fake_pos, fake_mask)
    print(logits.shape)

if __name__=='__main__':
    main()