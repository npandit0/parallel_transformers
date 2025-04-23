# Parallelizing Transformers over Depth

This is the repo for the [Stanford CS229S (Machine Learning Systems)](https://cs229s.stanford.edu/fall2024/) final project (Fall 2024) by Xavier Gonzalez and Nikhil Pandit. We use fixed-point iterations as developed by [DEER](https://arxiv.org/abs/2309.12252) and [ELK](https://arxiv.org/abs/2407.19115) to parallelize the application of transformer *layers*. Transformers are already parallel over the sequence length, but we parallelize them over depth, too! 

We demonstrate the capabilities of our approach on the pre-trained [Mistral-7B model](https://github.com/mistralai/mistral-src/tree/main?tab=readme-ov-file), [ported to JAX](https://github.com/AakashKumarNain/mistral_jax).

## Relevant Files
* `mistral_jax/deer_prototype_mistral.py`: implements inference with the Tranformer parallelized over depth.
* `mistral_jax/the_model_speaks.py`: implements autoregressive generation (using KV cache) but with the Transformer parallelized over depth.

## Citation

If you find this work useful, plese consider citing us with

```
@misc{GonzalezPanditParallelTransformers,
  title={Fully Parallelized Transformers: Parallelizing Transformers over Depth},
  author={Xavier Gonzalez and Nikhil Pandit},
  year={2025},
  url={https://github.com/xaviergonzalez/parallel_transformers/blob/main/parallel_transformers_paper.pdf},
}
```
