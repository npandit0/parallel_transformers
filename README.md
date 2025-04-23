# Parallelizing Transformers over Depth

This is the repo for the [Stanford CS229S (Machine Learning Systems)](https://cs229s.stanford.edu/fall2024/) final project (Fall 2024) by Xavier Gonzalez and Nikhil Pandit. We use fixed-point iterations as developed by [DEER](https://arxiv.org/abs/2309.12252) and [ELK](https://arxiv.org/abs/2407.19115) to parallelize the application of transformer *layers*. Transformers are already parallel over the sequence length, but we parallelize them over depth, too! 

We demonstrate the capabilities of our approach on the pre-trained [Mistral-7B model](https://github.com/mistralai/mistral-src/tree/main?tab=readme-ov-file), ported to JAX.
