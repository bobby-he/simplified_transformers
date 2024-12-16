# Simplified Transformers

This is the author's implementation for [Simplifying Transformer Blocks](https://arxiv.org/abs/2311.01906) (ICLR 2024) and [Understanding and Minimising Outlier Features in Transformer Training](https://arxiv.org/abs/2405.19279) (NeurIPS 2024).

<p align="center">
     <img src="assets/combined_blocks.png" width="600">
</p>


## Getting started
The main dependencies for this repo are:
- hydra
- wandb
- torch
- transformers (hf)
- datasets (hf)
- evaluate (hf)
- accelerate (hf)

To install these dependencies, run:  ```pip install -r requirements.txt```.
## Usage
This codebase runs the autoregressive experiments in our paper. The main training script is `run_clm.py`, which trains GPT-2 (small, ~120M params) on next-token prediction using code data, largely inspired by this HF [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter7/section6_pt.ipynb). It may take a few minutes to download the data on the first run.

We use [hydra](https://hydra.cc/docs/intro/) to organise our configs, so all arguments can be set from the command line. We assume training takes place on a single GPU.

The default config uses Pre-LN GPT-2, i.e. running:

```python run_clm.py num_token_mult=2 model.n_layer=18```

reproduces the Pre-LN run in Figure 2 of the paper, and should obtain eval loss of ~1.155 after 40K training steps. This takes ~10 hours on a A5000.

To change the model, we have 3 non-default configs set up from which you can make modifications: 
1. ```default-parallel``` (parallel block from [GPT-J](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/)), 
2. ```skipless``` (without attention sub-block skip, Figure 9 of paper)
3. ```skipless-parallel``` (parallel and skipless, Figure 10 of paper)

Other model settings can be customised from command line. For example, the following command reproduces the parallel, skipless block without normalisation (i.e. top right in header figure) in Figure 5:

```python run_clm.py num_token_mult=2 model.n_layer=18 model=skipless-parallel model.norm_type=none```

which should obtain eval loss of eval loss of ~1.245 after 40K steps. More training scripts can be found in ```exp_scripts/```.

We use [wandb](https://wandb.ai/) for logging by default. To turn this off, simply add ```use_wandb=False``` on command line.

## Outlier Feature computation
The kurtosis computation for outlier features can be found [here](https://github.com/bobby-he/simplified_transformers/blob/57137601c3b1d89b5f733065835c7f3a06b7d440/simplified_transformers/train_utils.py#L130). 

Note we take the variance (not second moment) of the normalised neuron-wise activation squared mean in [here](https://github.com/bobby-he/simplified_transformers/blob/57137601c3b1d89b5f733065835c7f3a06b7d440/simplified_transformers/train_utils.py#L147) which means we compute $kurtosis-1$. This is out by an additive constant of 1, but doesn't change our findings regarding preventing outlier features.

The config for the OP block is [here](https://github.com/bobby-he/simplified_transformers/blob/main/simplified_transformers/config/model/op.yaml).

## Citation
If you found this codebase useful, please consider citing:

```bib
@inproceedings{
he2024simplifying,
title={Simplifying Transformer Blocks},
author={Bobby He and Thomas Hofmann},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=RtDok9eS3s}
}
```

```bib
@inproceedings{
he2024understanding,
title={Understanding and Minimising Outlier Features in Transformer Training},
author={Bobby He and Lorenzo Noci and Daniele Paliotta and Imanol Schlag and Thomas Hofmann},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=npJQ6qS4bg}
}
```
