################## Visual Explainability for VAR Classification
# Generates 4-panel heatmap overlays showing which image regions
# drive classification decisions using per-token log-likelihood differences.
#
# This script operates in single-stage mode only (all 10 scales, no augmentation).
# For multi-stage classification, use eval.py instead.

import os
import os.path as osp
import torch
import random
import numpy as np
import logging
import sys
import time
import cv2

setattr(
    torch.nn.Linear, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed
setattr(
    torch.nn.LayerNorm, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed

from VAR_extensions import VQVAE, build_vae_var

import argparse
from datasets import build_dataset
from torch.utils.data import DataLoader, Subset
import tqdm
import json

import PIL.Image as PImage
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def make_json_serializable(obj):
    """
    Convert PyTorch tensors, NumPy types, and other non-serializable objects to JSON-serializable formats.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def compute_log_likelihood(logits, gt_tokens, vae, args, **kwargs):
    """
    Compute log probabilities using standard VAR likelihood.

    Args:
        logits: Model logits (B, L, V)
        gt_tokens: Ground truth tokens (B, L)
        vae: VAE model (unused for this score function)
        args: Arguments
        **kwargs: Additional arguments (unused for this score function)

    Returns:
        gt_log_probs: Log probabilities for ground truth tokens (B, L)
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    gt_log_probs = log_probs.gather(dim=-1, index=gt_tokens.unsqueeze(-1)).squeeze(-1)  # (B, L)
    return gt_log_probs


def load_imagenet_class_names(json_path="imagenet_class_index.json"):
    """
    Load ImageNet class names from JSON file.

    Args:
        json_path: Path to imagenet_class_index.json

    Returns:
        dict: Mapping from class_idx (int) -> class_name (str)
    """
    try:
        with open(json_path, 'r') as f:
            class_index = json.load(f)

        class_names = {}
        for idx_str, (synset, class_name) in class_index.items():
            class_names[int(idx_str)] = class_name.replace('_', ' ')

        return class_names
    except FileNotFoundError:
        logging.warning(f"Class index file not found at {json_path}. Using class indices as names.")
        return {}
    except Exception as e:
        logging.warning(f"Error loading class names: {e}. Using class indices as names.")
        return {}


def reshape_logprobs_to_multiscale(gt_log_probs, patch_nums):
    """
    Reshape flat log probabilities into multi-scale patches.

    Args:
        gt_log_probs: (seq_length,) - flat log probabilities
        patch_nums: tuple - patch numbers for each scale

    Returns:
        multiscale_logprobs: list of (patch_num, patch_num) arrays for each scale
    """
    multiscale_logprobs = []
    start_idx = 0

    for patch_num in patch_nums:
        tokens_this_scale = patch_num ** 2
        end_idx = start_idx + tokens_this_scale

        scale_tokens = gt_log_probs[start_idx:end_idx]
        scale_heatmap = scale_tokens.reshape(patch_num, patch_num)
        multiscale_logprobs.append(scale_heatmap)

        start_idx = end_idx

    return multiscale_logprobs


def create_weighted_heatmap(multiscale_logprobs, patch_nums, scale_weights):
    """
    Create weighted sum heatmap across all scales.

    Args:
        multiscale_logprobs: list of (patch_num, patch_num) arrays
        patch_nums: tuple - patch numbers for each scale
        scale_weights: list - weights for each scale

    Returns:
        weighted_heatmap: (patch_nums[-1], patch_nums[-1]) array
    """
    final_size = patch_nums[-1]
    weighted_heatmap = np.zeros((final_size, final_size))

    for scale_idx, (scale_heatmap, weight) in enumerate(zip(multiscale_logprobs, scale_weights)):
        if scale_heatmap.shape[0] != final_size:
            resized_heatmap = cv2.resize(scale_heatmap.astype(np.float32),
                                       (final_size, final_size),
                                       interpolation=cv2.INTER_NEAREST)
        else:
            resized_heatmap = scale_heatmap

        weighted_heatmap += weight * resized_heatmap

    return weighted_heatmap


def normalize_heatmap(heatmap):
    """
    Normalize heatmap to 0-1 range, clipping negative values.

    Args:
        heatmap: (H, W) array

    Returns:
        normalized_heatmap: (H, W) array in range [0, 1]
    """
    heatmap = np.clip(heatmap, a_min=0, a_max=None)

    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()

    if heatmap_max == heatmap_min:
        return np.zeros_like(heatmap)

    return (heatmap - heatmap_min) / (heatmap_max - heatmap_min)


def create_heatmap_overlay(original_image, heatmap, alpha=0.5):
    """
    Create heatmap overlay on original image using jet colormap.

    Args:
        original_image: PIL Image or numpy array (H, W, 3)
        heatmap: (H, W) normalized heatmap array
        alpha: float - overlay transparency

    Returns:
        overlay_image: PIL Image with heatmap overlay
    """
    if isinstance(original_image, PImage.Image):
        img_array = np.array(original_image)
    else:
        img_array = original_image.copy()

    img_h, img_w = img_array.shape[:2]
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (img_w, img_h))

    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)

    return PImage.fromarray(overlay)


def create_four_panel_visualization(original_image, heatmaps_dict, pred_label, true_label, y_neg_class, class_names=None):
    """
    Create 4-panel horizontal visualization layout with text overlays.

    Panels:
        1. Original input image
        2. logp(x|y) - logp(x) heatmap (true class vs unconditional)
        3. logp(x|y_neg) - logp(x) heatmap (best wrong class vs unconditional)
        4. logp(x|y) - logp(x|y_neg) or vice versa (conditional difference)

    Args:
        original_image: PIL Image
        heatmaps_dict: dict with keys 'cond_vs_uncond', 'neg_vs_uncond', 'cond_vs_neg'
        pred_label: int - predicted class
        true_label: int - true class
        y_neg_class: int - best wrong class
        class_names: dict - mapping from class_idx to class_name

    Returns:
        four_panel_image: PIL Image with 4 panels in horizontal layout
    """
    if class_names is None:
        class_names = {}

    true_class_name = class_names.get(true_label, str(true_label))
    neg_class_name = class_names.get(y_neg_class, str(y_neg_class))

    is_correct = (pred_label == true_label)

    panel1 = original_image
    panel2 = create_heatmap_overlay(original_image, heatmaps_dict['cond_vs_uncond'])
    panel3 = create_heatmap_overlay(original_image, heatmaps_dict['neg_vs_uncond'])
    panel4 = create_heatmap_overlay(original_image, heatmaps_dict['cond_vs_neg'])

    panels = [panel1, panel2, panel3, panel4]

    panel_texts = [
        "Input",
        rf"$\log\frac{{p(x|\text{{{true_class_name}}})}}{{p(x)}}$",
        rf"$\log\frac{{p(x|\text{{{neg_class_name}}})}}{{p(x)}}$",
        rf"$\log\frac{{p(x|\text{{{true_class_name if is_correct else neg_class_name}}})}}{{p(x|\text{{{neg_class_name if is_correct else true_class_name}}})}}$"
    ]

    panel_width, panel_height = panels[0].size
    combined_width = panel_width * 4
    combined_height = panel_height

    fig, axes = plt.subplots(1, 4, figsize=(combined_width/100, combined_height/100), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    for i, (ax, panel, text) in enumerate(zip(axes, panels, panel_texts)):
        panel_array = np.array(panel)
        ax.imshow(panel_array)
        ax.set_xlim(0, panel_width)
        ax.set_ylim(panel_height, 0)
        ax.axis('off')

        text_x = 4
        text_y = 4

        if text.startswith('$'):
            font_size = 10
        else:
            font_size = 8

        ax.text(text_x, text_y, text,
                fontsize=font_size, color='white', weight='bold',
                bbox=dict(boxstyle='square,pad=0.3', facecolor='black', alpha=0.8),
                verticalalignment='top', horizontalalignment='left',
                clip_on=True)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    plt.close(fig)

    combined_image = PImage.fromarray(buf)

    return combined_image


def extract_conditional_likelihoods(log_likelihood_tensor, per_token_log_probs_dict,
                                  label_batch, candidates_per_image):
    """
    Extract logp(x), logp(x|y), logp(x|y_neg) for each image in batch.

    Args:
        log_likelihood_tensor: (batch_size, num_candidates) - summed likelihoods (includes class 1000)
        per_token_log_probs_dict: dict mapping candidate_idx -> (batch_size, seq_length)
        label_batch: (batch_size,) - true labels
        candidates_per_image: list of candidate classes including 1000

    Returns:
        results: list of dicts with per-image conditional likelihoods and per-token probs
    """
    results = []
    current_batch_size = log_likelihood_tensor.shape[0]

    for i in range(current_batch_size):
        true_label = label_batch[i].item()

        # Extract unconditional likelihood (class 1000 is at the end)
        logp_x = log_likelihood_tensor[i, -1]
        unconditional_candidate_idx = len(candidates_per_image) - 1
        per_token_logp_x = per_token_log_probs_dict[unconditional_candidate_idx][i]

        # Extract true label likelihood
        try:
            true_label_idx = candidates_per_image.index(true_label)
            logp_x_given_y = log_likelihood_tensor[i, true_label_idx]
            per_token_logp_x_given_y = per_token_log_probs_dict[true_label_idx][i]
        except ValueError:
            logging.warning(f"True label {true_label} not in candidates for image {i}")
            continue

        # Find best wrong class (exclude true label and unconditional class 1000)
        valid_mask = torch.ones(len(candidates_per_image), dtype=torch.bool)
        valid_mask[true_label_idx] = False
        valid_mask[-1] = False  # Exclude unconditional class 1000

        if valid_mask.sum() == 0:
            logging.warning(f"No valid wrong candidates for image {i}")
            continue

        valid_likelihoods = log_likelihood_tensor[i, valid_mask]
        best_wrong_relative_idx = torch.argmax(valid_likelihoods)
        valid_indices = torch.where(valid_mask)[0]
        y_neg_idx = valid_indices[best_wrong_relative_idx]

        logp_x_given_y_neg = log_likelihood_tensor[i, y_neg_idx]
        per_token_logp_x_given_y_neg = per_token_log_probs_dict[y_neg_idx.item()][i]

        results.append({
            'logp_x': logp_x.item(),
            'logp_x_given_y': logp_x_given_y.item(),
            'logp_x_given_y_neg': logp_x_given_y_neg.item(),
            'per_token_logp_x': per_token_logp_x.detach().cpu().numpy(),
            'per_token_logp_x_given_y': per_token_logp_x_given_y.detach().cpu().numpy(),
            'per_token_logp_x_given_y_neg': per_token_logp_x_given_y_neg.detach().cpu().numpy(),
            'y_neg_class': candidates_per_image[y_neg_idx.item()],
            'image_idx': i
        })

    return results


BASE_LOG_DIR = "./outputs"

def main():
    parser = argparse.ArgumentParser(description="Visual explainability for VAR classification")

    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "imagenet-a", "imagenetv2", "imagenet-r", "imagenet-sketch", "objectnet"], help="Dataset to use")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--model_ckpt", type=str, default="./weights/imagenet/var_d16.pth", help="Path to VAR model checkpoint")
    parser.add_argument("--save_json", action='store_true', help="Save detailed JSON results for each sample")
    parser.add_argument("--synset_subset_path", type=str, default=None, help="Path to synset subset file (e.g., subsets/imagenet100.txt)")
    parser.add_argument("--sample_per_class", type=int, default=None, help="Maximum number of samples per class (ensures balanced sampling)")
    parser.add_argument("--extra", type=str, default="vis", help="Suffix after classification")
    parser.add_argument("--top_k_list", type=str, default="1,3,5,10,20,25,50,100", help="Comma-separated list of k values for top-k accuracy")
    args = parser.parse_args()

    # Parse k_list
    try:
        k_list = [int(x.strip()) for x in args.top_k_list.split(',')]
    except ValueError:
        raise ValueError(f"Invalid --top_k_list: '{args.top_k_list}'. Must be comma-separated integers.")
    if any(x < 1 for x in k_list):
        raise ValueError(f"All values in k_list must be >= 1: {k_list}")

    # Build vae, var
    if args.depth == 36:
        patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
    else:
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    patch_nums_square_cumsum = np.cumsum(np.array(patch_nums)**2)

    # All 10 scales, single-stage
    seq_length = patch_nums_square_cumsum[-1]

    name = "var"
    if args.depth != 16:
        name += f"_d{args.depth}"
    dataset_name = args.dataset
    if args.synset_subset_path:
        subset_name = os.path.basename(args.synset_subset_path).replace('.txt', '')
        dataset_name += f"_{subset_name}"
    if args.extra:
        LOG_DIR = BASE_LOG_DIR + f"_{args.extra}"

    score_func = "log_likelihood"
    run_folder = osp.join(LOG_DIR, dataset_name, score_func, name)
    os.makedirs(run_folder, exist_ok=True)

    # Create visualization output folders
    vis_success_folder = osp.join(run_folder, "vis_success")
    vis_failure_folder = osp.join(run_folder, "vis_failure")
    os.makedirs(vis_success_folder, exist_ok=True)
    os.makedirs(vis_failure_folder, exist_ok=True)

    # Setup logging
    log_file = osp.join(run_folder, "analysis.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Run folder: {run_folder}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Using batch size: {args.batch_size} (batching across both images and classes)")
    logging.info(f"Single-stage visualization mode: all {len(patch_nums)} scales, sequence length = {seq_length}")

    # Only create JSON output folders if saving JSON files
    if args.save_json:
        json_folder = osp.join(LOG_DIR, dataset_name, score_func, name, "json")
        os.makedirs(json_folder, exist_ok=True)

    # Load class names for visualization labels
    class_names = load_imagenet_class_names()
    logging.info(f"Loaded {len(class_names)} class names for visualization")

    # Build dataset
    data_path = "./datasets"
    dataset_val = build_dataset(
        data_path=data_path,
        final_reso=256 if args.depth != 36 else 512,
        dataset_type=args.dataset,
        synset_subset_path=args.synset_subset_path,
        sample_per_class=args.sample_per_class
    )

    num_classes = dataset_val.num_classes
    class_indices = dataset_val.subset_indices

    # Adjust batch size: candidates include all classes + class 1000
    num_candidates = len(class_indices) + 1  # +1 for unconditional class 1000
    adjusted_batch_size = max(args.batch_size // num_candidates, 1)

    # Create DataLoader
    total_samples_needed = len(dataset_val)
    ld_val = DataLoader(dataset_val, num_workers=0, pin_memory=True, batch_size=adjusted_batch_size, shuffle=False, drop_last=False)
    logging.info(f"Testing on whole dataset: {total_samples_needed} samples across {len(class_indices)} classes")
    logging.info(f"Adjusted batch size from {args.batch_size} to {adjusted_batch_size} to prevent OOM with {num_candidates} candidates (including unconditional)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    V = 4096
    logging.info(f"Building VAE and VAR-d{args.depth} model")
    vae, var_model = build_vae_var(
        V=V, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums, num_classes=1000, depth=args.depth, shared_aln=args.depth == 36
    )

    # Use model checkpoint from args
    model_ckpt = args.model_ckpt
    logging.info(f"Using model checkpoint: {model_ckpt}")

    # Download checkpoint if needed
    hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt = "vae_ch160v4096z32.pth"
    if not osp.exists(vae_ckpt):
        os.system(f"wget {hf_home}/{vae_ckpt}")
    if not osp.exists(model_ckpt):
        os.system(f"wget {hf_home}/{osp.basename(model_ckpt)}")

    # Load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)

    loaded_model_ckpt = torch.load(model_ckpt, map_location="cpu")
    if "trainer" in loaded_model_ckpt:
        trainer_state = loaded_model_ckpt["trainer"]
        if "var_wo_ddp" in trainer_state:
            loaded_model_ckpt = trainer_state["var_wo_ddp"]
            logging.info("Loading VAR model from 'trainer.var_wo_ddp' key (training checkpoint format)")
        else:
            loaded_model_ckpt = trainer_state
            logging.info("Loading VAR model from 'trainer' key")
    elif "var_wo_ddp" in loaded_model_ckpt:
        loaded_model_ckpt = loaded_model_ckpt["var_wo_ddp"]
        logging.info("Loading VAR model from 'var_wo_ddp' key (training checkpoint format)")
    else:
        logging.info("Loading VAR model directly (no nested structure)")
    var_model.load_state_dict(loaded_model_ckpt, strict=True)

    vae.eval(), var_model.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var_model.parameters():
        p.requires_grad_(False)

    logging.info("prepare finished.")
    var_model.cond_drop_rate = 0

    score_function = compute_log_likelihood
    logging.info("Using score function: log_likelihood")

    ############################# Classification with Visualization

    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

    correct = 0
    total = 0
    correct_dict = {}

    # Scale weights for heatmap aggregation
    scale_weights = [(pn / patch_nums[-1])**2 for pn in patch_nums]

    # Candidates: all classes + unconditional class 1000
    candidates_per_image = class_indices + [1000]

    start_time = time.time()
    logging.info("Starting classification with visualization...")

    pbar = tqdm.tqdm(ld_val, desc="Processing batches")
    for batch_idx, (img_batch, label_batch) in enumerate(pbar):
        if total > 0:
            pbar.set_description(f"Acc: {100 * correct / total:.2f}% ({total}/{total_samples_needed})")

        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)
        current_batch_size = img_batch.shape[0]

        with torch.inference_mode():
            batch_start_time = time.time()

            # Encode images through VQ-VAE
            f_original_batch = vae.img_to_post(img_batch)
            gt_idx_list_batch = vae.quantize.f_to_idxBl_or_fhat(f_original_batch, to_fhat=False, v_patch_nums=None)
            gt_tokens_batch = torch.cat(gt_idx_list_batch, dim=1)  # (current_batch_size, L)
            x_BLCv_wo_first_l_batch = vae.quantize.idxBl_to_var_input(gt_idx_list_batch)  # (current_batch_size, L, C)

            # Prepare data for all candidates
            current_candidates = torch.tensor([candidates_per_image] * current_batch_size, device=device)
            num_candidates_this_stage = current_candidates.shape[1]

            # Expand data for all (image, candidate) combinations
            candidates_flat = current_candidates.reshape(-1)
            x_BLCv_batch = x_BLCv_wo_first_l_batch[:, :seq_length-1, :].repeat_interleave(num_candidates_this_stage, dim=0)
            gt_tokens_expanded = gt_tokens_batch[:, :seq_length].repeat_interleave(num_candidates_this_stage, dim=0)

            num_total_combinations = current_batch_size * num_candidates_this_stage

            # Process in batches to avoid OOM, tracking per-token log probs
            batch_log_likelihood_list = []
            per_token_log_probs_dict = {}

            for batch_start in range(0, num_total_combinations, args.batch_size):
                batch_end = min(batch_start + args.batch_size, num_total_combinations)

                batch_candidates = candidates_flat[batch_start:batch_end]
                batch_x_BLCv = x_BLCv_batch[batch_start:batch_end]
                batch_gt_tokens = gt_tokens_expanded[batch_start:batch_end]

                logits = var_model.forward(batch_candidates, batch_x_BLCv)
                gt_log_probs = score_function(logits, batch_gt_tokens, vae, args)

                # Store per-token log probs for visualization
                for rel_idx in range(len(batch_candidates)):
                    abs_idx = batch_start + rel_idx
                    candidate_idx = abs_idx % len(candidates_per_image)
                    img_idx = abs_idx // len(candidates_per_image)

                    if candidate_idx not in per_token_log_probs_dict:
                        per_token_log_probs_dict[candidate_idx] = torch.zeros(current_batch_size, seq_length, device=device)
                    per_token_log_probs_dict[candidate_idx][img_idx] = gt_log_probs[rel_idx]

                log_likelihood = gt_log_probs.sum(dim=-1)
                batch_log_likelihood_list.append(log_likelihood)

            # Assemble full log-likelihood tensor (includes class 1000)
            all_log_likelihoods = torch.cat(batch_log_likelihood_list, dim=0)
            log_likelihood_tensor_with_1000 = all_log_likelihoods.reshape(current_batch_size, num_candidates_this_stage)

            # Remove class 1000 for prediction
            log_likelihood_tensor = log_likelihood_tensor_with_1000[:, :-1]

            # Predictions
            pred_indices = torch.argmax(log_likelihood_tensor, dim=1)
            pred_labels = torch.gather(current_candidates[:, :-1], 1, pred_indices.unsqueeze(1)).squeeze(1)

            batch_time = time.time() - batch_start_time

        # Top-k accuracy
        candidates_for_pred = current_candidates[:, :-1]  # Exclude class 1000
        if not correct_dict:
            max_candidates = candidates_for_pred.shape[1]
            valid_k_list = [k for k in k_list if k <= max_candidates]
            correct_dict = {k: 0 for k in valid_k_list}

        for k in valid_k_list:
            _, top_k_indices = torch.topk(log_likelihood_tensor, k=k, dim=1)
            top_k_candidates = torch.gather(candidates_for_pred, 1, top_k_indices)

            if args.dataset == "objectnet":
                for bi in range(current_batch_size):
                    objectnet_top_k = [dataset_val.class_idx_map.get(c.item(), -1) for c in top_k_candidates[bi]]
                    if label_batch[bi].item() in objectnet_top_k:
                        correct_dict[k] += 1
            else:
                is_correct_topk = (top_k_candidates == label_batch.unsqueeze(1)).any(dim=1)
                correct_dict[k] += is_correct_topk.sum().item()

        # Top-1 accuracy
        if args.dataset == "objectnet":
            objectnet_pred_labels = torch.tensor([dataset_val.class_idx_map.get(pred.item(), -1) for pred in pred_labels], device=device)
            batch_correct = (objectnet_pred_labels == label_batch).sum().item()
        else:
            batch_correct = (pred_labels == label_batch).sum().item()
        correct += batch_correct
        total += current_batch_size

        # Generate visualizations
        likelihood_results = extract_conditional_likelihoods(
            log_likelihood_tensor_with_1000, per_token_log_probs_dict,
            label_batch, candidates_per_image
        )

        for result in likelihood_results:
            img_idx = result['image_idx']
            true_label = label_batch[img_idx].item()
            pred_label = pred_labels[img_idx].item()
            y_neg_class = result['y_neg_class']

            # Denormalize image tensor to PIL
            original_img_tensor = img_batch[img_idx]
            img_array = original_img_tensor.permute(1, 2, 0).cpu().numpy()
            img_array = ((img_array + 1) * 127.5).clip(0, 255).astype(np.uint8)
            original_image = PImage.fromarray(img_array)

            # Compute three heatmaps
            heatmaps = {}

            # logp(x|y) - logp(x)
            diff_cond_vs_uncond = result['per_token_logp_x_given_y'] - result['per_token_logp_x']
            ms_cond_vs_uncond = reshape_logprobs_to_multiscale(diff_cond_vs_uncond, patch_nums)
            heatmaps['cond_vs_uncond'] = normalize_heatmap(create_weighted_heatmap(ms_cond_vs_uncond, patch_nums, scale_weights))

            # logp(x|y_neg) - logp(x)
            diff_neg_vs_uncond = result['per_token_logp_x_given_y_neg'] - result['per_token_logp_x']
            ms_neg_vs_uncond = reshape_logprobs_to_multiscale(diff_neg_vs_uncond, patch_nums)
            heatmaps['neg_vs_uncond'] = normalize_heatmap(create_weighted_heatmap(ms_neg_vs_uncond, patch_nums, scale_weights))

            # Conditional difference (direction depends on correctness)
            is_correct = (pred_label == true_label)
            if is_correct:
                diff_cond_vs_neg = result['per_token_logp_x_given_y'] - result['per_token_logp_x_given_y_neg']
            else:
                diff_cond_vs_neg = result['per_token_logp_x_given_y_neg'] - result['per_token_logp_x_given_y']
            ms_cond_vs_neg = reshape_logprobs_to_multiscale(diff_cond_vs_neg, patch_nums)
            heatmaps['cond_vs_neg'] = normalize_heatmap(create_weighted_heatmap(ms_cond_vs_neg, patch_nums, scale_weights))

            # Create and save 4-panel visualization
            four_panel_image = create_four_panel_visualization(
                original_image, heatmaps, pred_label, true_label, y_neg_class, class_names
            )

            global_idx = batch_idx * adjusted_batch_size + img_idx
            filename = f"{global_idx}_{true_label}_{pred_label}.png"

            if is_correct:
                save_path = osp.join(vis_success_folder, filename)
            else:
                save_path = osp.join(vis_failure_folder, filename)

            four_panel_image.save(save_path)

        # Save JSON files if requested
        if args.save_json:
            for img_idx in range(current_batch_size):
                label = label_batch[img_idx]
                pred = pred_labels[img_idx]

                if args.dataset == "objectnet":
                    pred_for_json = dataset_val.class_idx_map.get(pred.item(), -1)
                else:
                    pred_for_json = pred.item()

                global_idx = batch_idx * adjusted_batch_size + img_idx
                json_fname = osp.join(json_folder, f"{global_idx}.json")

                data = {
                    "pred": pred_for_json,
                    "label": label.item(),
                    f"pred_d{args.depth}": pred_for_json,
                    "metric_type": "log_likelihood",
                    "score_func": "log_likelihood",
                    "multi_stage": False,
                    "num_stages": 1,
                    "num_scale": len(patch_nums),
                    "sequence_length": int(seq_length),
                    "explanation": "Single-stage classification with visualization using log_likelihood score function."
                }

                json_safe_data = make_json_serializable(data)
                with open(json_fname, "w") as f:
                    json.dump(json_safe_data, f, indent=4)

    # Summary
    end_time = time.time()
    total_runtime = end_time - start_time
    logging.info(f"Classification completed in {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    logging.info(f"Average time per sample: {total_runtime/total:.3f} seconds")

    logging.info(f"\nOverall Accuracy (d{args.depth}, single-stage: scale={len(patch_nums)}): {100 * correct / total:.2f}%")

    if correct_dict:
        logging.info(f"\nTop-k Accuracies:")
        top_k_results = []
        for k in sorted(correct_dict.keys()):
            accuracy = 100 * correct_dict[k] / total
            top_k_results.append(f"Top-{k}: {accuracy:.2f}%")
        logging.info("  " + ", ".join(top_k_results))

    logging.info(f"\nVisualizations saved to:")
    logging.info(f"  Success: {vis_success_folder}")
    logging.info(f"  Failure: {vis_failure_folder}")


if __name__ == "__main__":
    main()
