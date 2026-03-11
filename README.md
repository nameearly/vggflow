# VGG-Flow

Official implementation of [Value Gradient Guidance for Flow Matching Alignment (VGG-Flow)](https://arxiv.org/abs/2512.05116), NeurIPS 2025.

<div align="center">
  <img src="assets/teaser.png" width="900"/>
</div>

## Overview

VGG-Flow is an efficient and robust RL finetuning method for flow-matching diffusion models.

This repository currently provides:

- SD3 LoRA finetuning with VGG-Flow.
- Multiple reward models (`aesthetic_score`, `pickscore`, `imagereward`, `hpscore`).
- Config-driven training with command-line overrides.

## Project Structure

- `train_vggflow.py`: main training entrypoint.
- `config/default_config.py`: base config values.
- `config/*.py`: reward-specific experiment configs.
- `lib/`: model, diffusion, reward, and training modules.
- `run.sh`: example launch command with many overrides.

## Setup

### Requirements

- Python >= 3.8
- CUDA (tested with 12.4)
- PyTorch + TorchVision

Install dependencies:

```bash
pip install -r requirements.txt
```

### Model Access

By default, training loads `stabilityai/stable-diffusion-3-medium-diffusers` via Diffusers.
Make sure your environment has access to required Hugging Face model weights.

## Configuration

Before training, review `config/default_config.py` and update values as needed.

Important fields:

- `config.logging.wandb_key`: replace `"PLACEHOLDER"` if using Weights & Biases.
- `config.logging.use_wandb`: set to `False` to disable W&B logging.
- `config.logging.wandb_dir`: local W&B output directory.
- `config.saving.output_dir`: checkpoint/output directory.

Reward presets:

- `config/aesthetic.py`
- `config/pickscore.py`
- `config/imagereward.py`
- `config/hpsv2.py`

## Quick Start

Example: 2-GPU single-node training with the aesthetic reward preset:

```bash
torchrun --standalone --nproc_per_node=2 train_vggflow.py \
  --config=config/aesthetic.py \
  --seed=1 \
  --exp_name=exp_aesthetic
```

Single-GPU run:

```bash
torchrun --standalone --nproc_per_node=1 train_vggflow.py \
  --config=config/aesthetic.py \
  --seed=1 \
  --exp_name=exp_aesthetic
```

Override any config value from the command line, for example:

```bash
torchrun --standalone --nproc_per_node=2 train_vggflow.py \
  --config=config/aesthetic.py \
  --config.model.reward_scale=1e4 \
  --config.sampling.num_steps=20 \
  --config.training.lr=1e-3 \
  --seed=1 \
  --exp_name=exp_custom
```

## Key Hyperparameters

- `config.model.reward_scale`: reward strength; larger values push harder toward the reward objective.
- `config.model.timestep_fraction`: fraction of trajectory transitions used for updates.
- `config.sampling.num_steps`: number of diffusion sampling steps.
- `config.training.lr`: optimizer learning rate.
- `config.training.batch_size` + `config.training.gradient_accumulation_steps`: effective optimization batch size.
- `config.model.unet_reg_scale`: regularization strength for preserving base behavior.

## Outputs

- Checkpoints are saved under:
  - `config.saving.output_dir/<reward>_vggflow_<exp_name>_seed<seed>/checkpoint_epoch*`
- Training stats are written to:
  - `.../result.json` (compressed pickle format)
- If enabled, metrics and sample images are also logged to W&B.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{liu2025vggflow,
  title={Value Gradient Guidance for Flow Matching Alignment},
  author={Liu, Zhen and Xiao, Tim Z. and Liu, Weiyang and Domingo-Enrich, Carles and Zhang, Dinghuai},
  booktitle={NeurIPS},
  year={2025},
}
```

