# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**A-VARC** (Autoregressive Visual AutoRegressive Classifier) is a research codebase for image classification using Visual Autoregressive (VAR) models. It builds on the VAR framework (NeurIPS 2024 Best Paper) to perform classification by computing log-likelihoods of images conditioned on class labels.

The project consists of:
- **VAR submodule**: The upstream VAR model implementation (from FoundationVision/VAR)
- **var_classification.py**: Main classification script that uses VAR models as generative classifiers

### Key Concept

Unlike traditional discriminative classifiers, this approach treats classification as a generative modeling task. For each candidate class, it computes the log-likelihood of the input image given that class using the VAR model's autoregressive next-scale prediction. The class with highest likelihood is the prediction.

## Architecture

### VAR Model Structure

The VAR (Visual Autoregressive) model performs "next-scale prediction" rather than traditional "next-token prediction":
- Input images are encoded by a VQ-VAE into discrete tokens at multiple resolution scales
- The model autoregressively predicts tokens from coarse-to-fine (scale 1→2→3...→10)
- **patch_nums**: `(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)` defines the 10 resolution scales
- Each scale i has `patch_nums[i]²` tokens (e.g., scale 0 has 1×1=1 token, scale 9 has 16×16=256 tokens)
- Total sequence length L = sum of all tokens across scales

### Classification Pipeline

**var_classification.py** implements multi-stage classification:

1. **Image Encoding**: VQ-VAE encodes input image → discrete tokens at all scales
2. **Stage-by-stage Filtering**:
   - Each stage uses a subset of scales (controlled by `num_scale_list`)
   - Computes log-likelihoods for candidate classes
   - Filters to top-k candidates for next stage (controlled by `num_candidate_list`)
3. **Sample Augmentation** (optional): Generate noisy neighbors of input image and average likelihoods
4. **Final Prediction**: Class with highest log-likelihood in final stage

### Model Variants

- **VAR-d16** (310M params): depth=16, 256×256 images
- **VAR-d20** (600M params): depth=20, 256×256 images
- **VAR-d24** (1.0B params): depth=24, 256×256 images
- **VAR-d30** (2.0B params): depth=30, 256×256 images
- **VAR-d36** (2.3B params): depth=36, 512×512 images, uses shared AdaLN

### Key Components

**VAR/models/**:
- `vqvae.py`: VQ-VAE encoder/decoder with multi-scale quantization
- `var.py`: Core VAR transformer with AdaLN conditioning
- `basic_var.py`: AdaLN attention blocks with optional flash-attention
- `quant.py`: Vector quantization with residual quantization

**VAR/utils/**:
- `data.py`: Dataset loading for ImageNet and variants
- `arg_util.py`: Training argument parsing
- `misc.py`: Auto-resume, logging utilities

**Classification-specific**:
- `var_classification.py`: Standalone classification script with multi-stage support

## Common Commands

### Classification

Run classification on ImageNet validation set:
```bash
python var_classification.py \
  --dataset imagenet \
  --depth 16 \
  --batch_size 1 \
  --score_func var
```

Multi-stage classification with progressive filtering:
```bash
python var_classification.py \
  --dataset imagenet \
  --depth 16 \
  --num_candidate_list "1000,100,10,1" \
  --num_sample_list "1,1,3,5" \
  --num_scale_list "3,6,8,10" \
  --save_json
```

Using LoRA fine-tuned weights:
```bash
python var_classification.py \
  --depth 16 \
  --lora_weights noise_aug \
  --dataset imagenet
```

### Training (VAR submodule)

Train VAR model on ImageNet:
```bash
cd VAR
torchrun --nproc_per_node=8 train.py \
  --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1
```

### Dataset Setup

ImageNet dataset structure expected at `./datasets/`:
```
datasets/imagenet/
  train/
    n01440764/
      many_images.JPEG ...
  val/
    n01440764/
      ILSVRC2012_val_00000293.JPEG ...
```

## Important Implementation Details

### Score Functions

The `--score_func` argument selects the likelihood computation method (currently only `var` is implemented):
- **var**: Standard VAR log-likelihood using cross-entropy loss
- Additional score functions can be registered in `SCORE_FUNCTIONS` dict

### Multi-Stage Classification

Three parallel lists control the staged classification:
- `num_candidate_list`: Number of candidates per stage (must be decreasing, end with 1)
- `num_sample_list`: Number of augmented samples per stage (≥1)
- `num_scale_list`: Number of VAR scales to use per stage (1-10)

Example: `[1000,100,1]` candidates, `[1,1,5]` samples, `[3,6,10]` scales
- Stage 0: 1000 classes, 1 sample, 3 scales (quick filtering)
- Stage 1: 100 classes, 1 sample, 6 scales (medium filtering)
- Stage 2: 1 class, 5 samples, 10 scales (final prediction with augmentation)

### LoRA Model Loading

When using `--lora_weights`:
- Expects checkpoint at `./lora_weights/var_d{depth}_{lora_weights}/ar-ckpt-best.pth`
- Requires PEFT library for proper LoRA merging (`pip install peft`)
- Falls back to base weights only if PEFT unavailable

### Output Formats

With `--save_json`: Saves detailed per-image results to `classification_{extra}/{dataset}/{score_func}/{name}/json/{idx}.json`
- Includes per-stage log-likelihoods, candidates, and predictions
- Contains configuration metadata for reproducibility

With `--save_likelihood`: Saves per-token log-probabilities as `likelihood/{idx}.npz`
- Only supported for single-stage classification
- Format: `log_likelihood` (N, L), `candidates` (N,), `gt_label`, `sequence_length`

## Common Gotchas

1. **Checkpoint Format Variations**: The code handles multiple checkpoint formats:
   - Standard VAR checkpoints: direct state dict
   - Training checkpoints: nested under `trainer.var_wo_ddp`
   - LoRA checkpoints: PEFT wrapped with `base_model.model.` prefix

2. **Batch Size with Many Classes**: Effective batch size is `batch_size // test_num_classes` to prevent OOM when testing many classes simultaneously

3. **ObjectNet Evaluation**: Requires special handling with `class_idx_map` to translate ImageNet predictions to ObjectNet class space

4. **Sequence Length**: Different stages use different sequence lengths based on `num_scale_list[i]`. Length = `cumsum(patch_nums²)[num_scale_list[i]-1]`

5. **Sample Augmentation**: When `num_sample_list[i] > 1`, adds Gaussian noise (variance=`sigma`) to latent features and averages likelihoods

## Dependencies

Core requirements (see VAR/requirements.txt):
- PyTorch >= 2.1.0
- Pillow, numpy, transformers
- huggingface_hub

Optional for performance:
- flash-attn (for faster attention)
- xformers (alternative fast attention)
- peft (for LoRA fine-tuning support)

Additional datasets require installing their specific packages.
