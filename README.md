# Your VAR Model is Secretly an Efficient and Explainable Generative Classifier

[Paper](https://openreview.net/pdf?id=zFF0WRMjlp)

**Authors:** Yi-Chung Chen, David I. Inouye, Jing Gao — Purdue University

![Banner](assets/banner.jpg)

## Overview

We investigate generative classifiers built upon recent advances in visual autoregressive (VAR) modeling. Owing to their tractable likelihood, VAR-based generative classifiers enable significantly more efficient inference compared to diffusion-based counterparts. Building on this foundation, we introduce the **Adaptive VAR Classifier+ (A-VARC+)**, which further improves accuracy while reducing computational cost, substantially enhancing practical usability.

## Installation

Coming soon.

## Quick Start

Run classification on ImageNet validation set:

```bash
python eval.py \
  --dataset imagenet \
  --depth 16 \
  --batch_size 1 \
  --model_ckpt ./weights/imagenet/var_d16.pth \
  --num_candidate_list "10,3,1" \
  --num_sample_list "1,1,3" \
  --num_scale_list "6,10,10" \
  --synset_subset_path subsets/imagenet100.txt
```

## Comparative Analysis

![Comparative Analysis](assets/Comparative_analysis.png)

## Class-Incremental Learning

![Class-Incremental Learning](assets/Class_incremental_learning.png)

## Citation

```bibtex
@article{chen2025your,
  title={Your VAR Model is Secretly an Efficient and Explainable Generative Classifier},
  author={Chen, Yi-Chung and Inouye, David I and Gao, Jing},
  journal={arXiv preprint arXiv:2510.12060},
  year={2025}
}
```

## Acknowledgements

This project builds on the [VAR](https://github.com/FoundationVision/VAR) codebase.

## License

See [LICENSE](LICENSE) for details.
