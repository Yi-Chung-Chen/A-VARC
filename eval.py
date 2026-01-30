################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch
import random
import numpy as np
import logging
import sys
import time

setattr(
    torch.nn.Linear, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed
setattr(
    torch.nn.LayerNorm, "reset_parameters", lambda self: None
)  # disable default parameter init for faster speed

# Import from VAR_extensions which includes custom modifications
from VAR_extensions import VQVAE, build_vae_var

import argparse
from datasets import build_dataset
from torch.utils.data import DataLoader, Subset
import tqdm
import json

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


MODEL_DEPTH = 16  # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}
BASE_LOG_DIR = "./outputs"

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

# Score function is hardcoded to log_likelihood

def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "imagenet-a", "imagenetv2", "imagenet-r", "imagenet-sketch", "objectnet"], help="Dataset to use")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--num_candidate_list", type=str, default="1", help="Comma-separated list of candidates surviving each stage (e.g., '1000,100,10,1'). Must be decreasing with last value = 1.")
    parser.add_argument("--num_sample_list", type=str, default="1", help="Comma-separated list of samples for each stage (e.g., '1,1,3,5'). Each value >= 1.")
    parser.add_argument("--num_scale_list", type=str, default="10", help="Comma-separated list of scales for each stage (e.g., '3,6,8,10'). Each value in [1, len(patch_nums)].")
    parser.add_argument("--model_ckpt", type=str, default="./weights/imagenet/var_d16.pth", help="Path to VAR model checkpoint")
    parser.add_argument("--save_json", action='store_true', help="Save detailed JSON results for each sample")
    parser.add_argument("--sigma", type=float, default=0.1, help="Variance of the Gaussian noise added to the neighbors")
    parser.add_argument("--synset_subset_path", type=str, default=None, help="Path to synset subset file (e.g., imagenet100.txt)")
    parser.add_argument("--sample_per_class", type=int, default=None, help="Maximum number of samples per class (ensures balanced sampling)")
    parser.add_argument("--extra", type=str, default="formal", help="Suffix after classification")
    parser.add_argument("--top_k_list", type=str, default="1,3,5,10,20,25,50,100", help="Comma-separated list of k values for top-k accuracy (e.g., '1,3,5,10')")
    args = parser.parse_args()
    MODEL_DEPTH = args.depth

    # Parse and validate list arguments
    def parse_int_list(list_str, arg_name):
        try:
            return [int(x.strip()) for x in list_str.split(',')]
        except ValueError:
            raise ValueError(f"Invalid {arg_name}: '{list_str}'. Must be comma-separated integers.")

    num_candidate_list = parse_int_list(args.num_candidate_list, "--num_candidate_list")
    num_sample_list = parse_int_list(args.num_sample_list, "--num_sample_list")
    num_scale_list = parse_int_list(args.num_scale_list, "--num_scale_list")
    k_list = parse_int_list(args.top_k_list, "--top_k_list")

    # build vae, var
    if args.depth == 36:
        patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
    else:
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    patch_nums_square_cumsum = np.cumsum(np.array(patch_nums)**2)

    # Validate multi-stage arguments
    num_stages = len(num_candidate_list)
    if len(num_sample_list) != num_stages or len(num_scale_list) != num_stages:
        raise ValueError(f"All lists must have the same length. Got: num_candidate_list={len(num_candidate_list)}, num_sample_list={len(num_sample_list)}, num_scale_list={len(num_scale_list)}")

    # Validate num_candidate_list: decreasing, last value = 1
    if num_candidate_list != sorted(num_candidate_list, reverse=True):
        raise ValueError(f"num_candidate_list must be decreasing: {num_candidate_list}")
    if num_candidate_list[-1] != 1:
        raise ValueError(f"Last value in num_candidate_list must be 1, got: {num_candidate_list[-1]}")

    # Validate num_sample_list: all values >= 1
    if any(x < 1 for x in num_sample_list):
        raise ValueError(f"All values in num_sample_list must be >= 1: {num_sample_list}")

    # Validate num_scale_list: all values in [1, len(patch_nums)]
    if any(x < 1 or x > len(patch_nums) for x in num_scale_list):
        raise ValueError(f"All values in num_scale_list must be in [1, {len(patch_nums)}]: {num_scale_list}")

    # Validate k_list: all values >= 1
    if any(x < 1 for x in k_list):
        raise ValueError(f"All values in k_list must be >= 1: {k_list}")

    name = f"var"
    if args.depth != 16:
        name += f"_d{args.depth}"

    # Add multi-stage configuration to name if not default
    default_candidates = [1]
    default_samples = [1]
    default_scales = [10]

    if num_candidate_list != default_candidates:
        name += f"_candidates[{','.join(map(str, num_candidate_list))}]"
    if num_sample_list != default_samples:
        name += f"_samples[{','.join(map(str, num_sample_list))}]"
    if num_scale_list != default_scales:
        name += f"_scales[{','.join(map(str, num_scale_list))}]"

    if max(num_sample_list) > 1:
        name += f"_sigma{args.sigma}"
    dataset_name = args.dataset
    if args.synset_subset_path:
        subset_name = os.path.basename(args.synset_subset_path).replace('.txt', '')
        dataset_name += f"_{subset_name}"
    if args.extra:
        LOG_DIR = BASE_LOG_DIR + f"_{args.extra}"

    # Hardcoded score_func to "log_likelihood"
    score_func = "log_likelihood"
    run_folder = osp.join(LOG_DIR, dataset_name, score_func, name)
    os.makedirs(run_folder, exist_ok=True)
    
    # Setup standard logging instead of the custom PrintLogger
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

    # Log multi-stage configuration
    logging.info(f"Multi-stage classification configuration:")
    logging.info(f"  Number of stages: {num_stages}")
    for stage_idx in range(num_stages):
        stage_num_candidates = num_candidate_list[stage_idx]
        stage_num_samples = num_sample_list[stage_idx]
        stage_num_scales = num_scale_list[stage_idx]
        clf_layer = stage_num_scales - 1
        stage_patch_size = patch_nums[clf_layer]
        stage_seq_length = patch_nums_square_cumsum[clf_layer]

        logging.info(f"  Stage {stage_idx}: candidates={stage_num_candidates}, samples={stage_num_samples}, scales={stage_num_scales}")
        logging.info(f"    Patch size: {stage_patch_size}, Sequence length: {stage_seq_length} tokens")

    # Log sample augmentation information
    if max(num_sample_list) > 1:
        logging.info(f"Sample augmentation enabled:")
        max_samples = max(num_sample_list)
        logging.info(f"  Maximum samples per image: {max_samples} (1 original + {max_samples-1} neighbors)")
        logging.info(f"  Gaussian noise variance: {args.sigma}")
        stages_with_augmentation = [i for i, s in enumerate(num_sample_list) if s > 1]
        if stages_with_augmentation:
            logging.info(f"  Applied to stages: {stages_with_augmentation}")
        else:
            logging.info(f"  Applied to: No stages (all stages use 1 sample)")

    # Only create JSON output folders if saving JSON files
    if args.save_json:
        json_folder = osp.join(LOG_DIR, dataset_name, score_func, name, "json")
        os.makedirs(json_folder, exist_ok=True)

    # Build dataset
    data_path = f"./datasets"
    dataset_val = build_dataset(
        data_path=data_path,
        final_reso=256 if args.depth != 36 else 512,
        dataset_type=args.dataset,
        synset_subset_path=args.synset_subset_path,
        sample_per_class=args.sample_per_class
    )
    
    # Extract information from dataset attributes
    num_classes = dataset_val.num_classes
    class_indices = dataset_val.subset_indices  # Use subset_indices which contains filtered indices or falls back to class_indices

    # Adjust batch size to prevent OOM when testing many classes
    adjusted_batch_size = max(args.batch_size // len(class_indices), 1)

    # Create DataLoader
    total_samples_needed = len(dataset_val)
    ld_val = DataLoader(dataset_val, num_workers=0, pin_memory=True, batch_size=adjusted_batch_size, shuffle=False, drop_last=False)
    logging.info(f"Testing on whole dataset: {total_samples_needed} samples across {len(class_indices)} classes")
    logging.info(f"Adjusted batch size from {args.batch_size} to {adjusted_batch_size} to prevent OOM with {len(class_indices)} classes")
    
    
    # Validate num_scale range
    if min(num_scale_list) < 1 or max(num_scale_list) > len(patch_nums):
        raise ValueError(f"value of num_scale_list must be between 1 and {len(patch_nums)}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    V = 4096
    # Build the model
    logging.info(f"Building VAE and VAR-d{args.depth} model")
    vae, var_model = build_vae_var(
        V=V, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums, num_classes=1000, depth=args.depth, shared_aln=args.depth == 36
    )

    # Use model checkpoint from args
    model_ckpt = args.model_ckpt
    logging.info(f"Using model checkpoint: {model_ckpt}")

    # download checkpoint
    hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
    vae_ckpt = "vae_ch160v4096z32.pth"
    if not osp.exists(vae_ckpt):
        os.system(f"wget {hf_home}/{vae_ckpt}")
    if not osp.exists(model_ckpt):
        os.system(f"wget {hf_home}/{osp.basename(model_ckpt)}")

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)

    # Standard model loading
    loaded_model_ckpt = torch.load(model_ckpt, map_location="cpu")
    # Handle different checkpoint formats
    if "trainer" in loaded_model_ckpt:
        trainer_state = loaded_model_ckpt["trainer"]

        if "var_wo_ddp" in trainer_state:
            loaded_model_ckpt = trainer_state["var_wo_ddp"]
            logging.info("Loading VAR model from 'trainer.var_wo_ddp' key (training checkpoint format)")
        else:
            loaded_model_ckpt = trainer_state
            logging.info("Loading VAR model from 'trainer' key")
    elif "var_wo_ddp" in loaded_model_ckpt:
        # Handle training checkpoint format
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

    # Hardcode score function to log_likelihood
    score_function = compute_log_likelihood
    logging.info("Using score function: log_likelihood")

    ############################# 2. Sample with classifier-free guidance

    seed = 0
    torch.manual_seed(seed)
    num_sampling_steps = 250
    more_smooth = False

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
    correct_dict = {}  # Will be initialized when we know valid_k_list

    # Start timing
    start_time = time.time()
    stage_total_times = [0.0] * num_stages
    logging.info("Starting classification...")

    pbar = tqdm.tqdm(ld_val, desc="Processing batches")
    for batch_idx, (img_batch, label_batch) in enumerate(pbar):
        if total > 0:
            pbar.set_description(f"Acc: {100 * correct / total:.2f}% ({total}/{total_samples_needed})")
        
        # Skip computation if output files (JSON) already exist for this batch
        if args.save_json:
            current_batch_size = img_batch.shape[0]
            skip_batch = True
            for img_idx in range(current_batch_size):
                global_idx = batch_idx * adjusted_batch_size + img_idx
                json_fname = osp.join(json_folder, f"{global_idx}.json")
                if not osp.exists(json_fname):
                    skip_batch = False
                    break

            if skip_batch:
                # All output files exist, skip computation but load predictions to update accuracy
                img_batch = img_batch.to(device)
                label_batch = label_batch.to(device)
                batch_correct = 0
                for img_idx in range(current_batch_size):
                    global_idx = batch_idx * adjusted_batch_size + img_idx
                    json_fname = osp.join(json_folder, f"{global_idx}.json")
                    with open(json_fname, 'r') as f:
                        data = json.load(f)
                    pred = data['pred']
                    label = label_batch[img_idx].item()
                    if pred == label:
                        batch_correct += 1

                correct += batch_correct
                total += current_batch_size
                logging.info(f"Skipped batch {batch_idx} (output files already exist)")
                continue
        
        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)
        current_batch_size = img_batch.shape[0]  # Handle last batch which might be smaller
        
        # List of classes to process for this sample.
        # Use all available class indices
        test_classes = class_indices
        
        with torch.inference_mode():
            batch_start_time = time.time()

            # Process images with single crop
            f_original_batch = vae.img_to_post(img_batch)

            gt_idx_list_batch = vae.quantize.f_to_idxBl_or_fhat(f_original_batch, to_fhat=False, v_patch_nums=None)
            gt_tokens_batch_full = torch.cat(gt_idx_list_batch, dim=1)  # (current_batch_size, L_full) - Keep full length for all stages
            x_BLCv_wo_first_l_batch_full = vae.quantize.idxBl_to_var_input(gt_idx_list_batch)  # (current_batch_size, L_full, C) - Keep full length for all stages

            # Initialize variables for multi-stage processing
            # Create per-image candidate tensor: (current_batch_size, num_test_classes)
            current_candidates = torch.tensor([test_classes] * current_batch_size, device=device)
            stage_results = []  # Store results for each stage
            final_pred_labels = None
            top_k_data = None  # Will be set in final stage

            # Multi-stage classification loop
            for stage_idx in range(num_stages):
                stage_start_time = time.time()

                # Get stage configuration
                stage_num_candidates = num_candidate_list[stage_idx]
                stage_num_samples = num_sample_list[stage_idx]
                stage_num_scales = num_scale_list[stage_idx]

                # Calculate sequence length for this stage
                clf_layer = stage_num_scales - 1
                seq_length = patch_nums_square_cumsum[clf_layer]

                # Prepare data for this stage
                x_BLCv_wo_first_l_batch = x_BLCv_wo_first_l_batch_full[:, :seq_length-1, :]
                gt_tokens_batch = gt_tokens_batch_full[:, :seq_length]

                # Generate neighbor data if sample augmentation is enabled for this stage
                neighbor_x_BLCv_wo_first_l_batch = None
                neighbor_tokens_batch = None
                if stage_num_samples > 1:
                    num_neighbor = stage_num_samples - 1
                    # Generate neighbor features by adding noise to the original features
                    neighbor_f_batch = f_original_batch.repeat(num_neighbor, 1, 1, 1)  # (current_batch_size * num_neighbor, C, H, W)
                    neighbor_noise = torch.randn_like(neighbor_f_batch) * args.sigma
                    neighbor_f_batch = neighbor_f_batch + neighbor_noise

                    # Get neighbor tokens
                    neighbor_idx_list_batch = vae.quantize.f_to_idxBl_or_fhat(neighbor_f_batch, to_fhat=False, v_patch_nums=None)
                    neighbor_tokens_batch_full = torch.cat(neighbor_idx_list_batch, dim=1)  # (current_batch_size * num_neighbor, L_full)
                    neighbor_x_BLCv_wo_first_l_batch_full = vae.quantize.idxBl_to_var_input(neighbor_idx_list_batch)  # (current_batch_size * num_neighbor, L_full, C)

                    # Trim to current sequence length
                    neighbor_x_BLCv_wo_first_l_batch = neighbor_x_BLCv_wo_first_l_batch_full[:, :seq_length-1, :]
                    neighbor_tokens_batch = neighbor_tokens_batch_full[:, :seq_length]

                # Process classes for this stage with per-image candidates
                # current_candidates shape: (current_batch_size, num_candidates_this_stage)
                num_candidates_this_stage = current_candidates.shape[1]

                # Create all (image, candidate) combinations for batch processing
                # Expand candidates: (current_batch_size, num_candidates_this_stage) -> (current_batch_size * num_candidates_this_stage,)
                candidates_flat = current_candidates.reshape(-1)

                # Expand image data for all candidates
                # Shape: (current_batch_size * num_candidates_this_stage, seq_length-1, C)
                x_BLCv_batch = x_BLCv_wo_first_l_batch.repeat_interleave(num_candidates_this_stage, dim=0)

                # Expand ground truth tokens for all candidates
                # Shape: (current_batch_size * num_candidates_this_stage, seq_length)
                gt_tokens_expanded = gt_tokens_batch.repeat_interleave(num_candidates_this_stage, dim=0)

                # Combine original and neighbor data first, then batch process
                num_total_combinations = current_batch_size * num_candidates_this_stage

                if stage_num_samples > 1:
                    # Expand neighbor data to match candidate structure
                    neighbor_x_BLCv_batch = neighbor_x_BLCv_wo_first_l_batch.repeat_interleave(num_candidates_this_stage, dim=0)
                    neighbor_tokens_expanded = neighbor_tokens_batch.repeat_interleave(num_candidates_this_stage, dim=0)
                    neighbor_candidates_flat = candidates_flat.repeat(num_neighbor)

                    # Combine original and neighbor data
                    combined_x_BLCv_batch = torch.cat([x_BLCv_batch, neighbor_x_BLCv_batch], dim=0)
                    combined_tokens_expanded = torch.cat([gt_tokens_expanded, neighbor_tokens_expanded], dim=0)
                    combined_candidates_flat = torch.cat([candidates_flat, neighbor_candidates_flat], dim=0)

                    num_total_combinations_with_samples = num_total_combinations * stage_num_samples
                else:
                    # No neighbors, use original data
                    combined_x_BLCv_batch = x_BLCv_batch
                    combined_tokens_expanded = gt_tokens_expanded
                    combined_candidates_flat = candidates_flat
                    num_total_combinations_with_samples = num_total_combinations

                # Process in batches to avoid OOM
                batch_log_likelihood_list = []

                for batch_start in range(0, num_total_combinations_with_samples, args.batch_size):
                    batch_end = min(batch_start + args.batch_size, num_total_combinations_with_samples)

                    # Get batch from combined data
                    batch_candidates = combined_candidates_flat[batch_start:batch_end]
                    batch_x_BLCv = combined_x_BLCv_batch[batch_start:batch_end]
                    batch_gt_tokens = combined_tokens_expanded[batch_start:batch_end]

                    # Single forward pass (no need for separate neighbor handling)
                    logits = var_model.forward(batch_candidates, batch_x_BLCv)
                    gt_log_probs = score_function(logits, batch_gt_tokens, vae, args)

                    # Calculate overall log likelihood
                    log_likelihood = gt_log_probs.sum(dim=-1)  # (batch_size_current,)
                    batch_log_likelihood_list.append(log_likelihood)

                # Concatenate all batch results
                all_log_likelihoods = torch.cat(batch_log_likelihood_list, dim=0)

                # Handle sample averaging if neighbors were used
                if stage_num_samples > 1:
                    # Reshape: (stage_num_samples * num_total_combinations,) -> (stage_num_samples, num_total_combinations)
                    all_log_likelihoods = all_log_likelihoods.reshape(stage_num_samples, num_total_combinations)
                    # Average over samples: (stage_num_samples, num_total_combinations) -> (num_total_combinations,)
                    all_log_likelihoods = all_log_likelihoods.mean(dim=0)

                # Final reshape to (current_batch_size, num_candidates_this_stage)
                stage_log_likelihood_tensor = all_log_likelihoods.reshape(current_batch_size, num_candidates_this_stage)

                # Store stage results
                stage_end_time = time.time()
                stage_batch_time = stage_end_time - stage_start_time
                stage_total_times[stage_idx] += stage_batch_time

                stage_result = {
                    'stage': int(stage_idx),
                    'num_candidates': int(stage_num_candidates),
                    'num_samples': int(stage_num_samples),
                    'num_scales': int(stage_num_scales),
                    'sequence_length': int(seq_length),
                    'log_likelihood': stage_log_likelihood_tensor.detach().cpu(),  # Keep as tensor for processing, convert later
                    'candidates': current_candidates.cpu().tolist(),  # Store per-image candidates: (batch_size, num_candidates)
                    'processing_time': float(stage_batch_time)
                }

                # Final predictions if this is the last stage
                if stage_idx == num_stages - 1:
                    # Last stage: get final predictions
                    pred_indices = torch.argmax(stage_log_likelihood_tensor, dim=1)  # (current_batch_size,)
                    # Use gather to get final predictions for each image
                    final_pred_labels = torch.gather(current_candidates, 1, pred_indices.unsqueeze(1)).squeeze(1)  # (current_batch_size,)
                    stage_result['final_prediction'] = True

                    # Prepare top-k accuracy data for later use
                    max_candidates = current_candidates.shape[1]
                    valid_k_list = [k for k in k_list if k <= max_candidates]
                    top_k_data = {
                        'valid_k_list': valid_k_list,
                        'stage_log_likelihood_tensor': stage_log_likelihood_tensor,
                        'current_candidates': current_candidates
                    }
                else:
                    # Not the last stage: filter to top candidates for next stage
                    _, top_indices = torch.topk(stage_log_likelihood_tensor, k=num_candidate_list[stage_idx], dim=1)  # (current_batch_size, k)

                    # Extract top candidates for next stage using proper tensor operations
                    # current_candidates: (current_batch_size, num_candidates_this_stage)
                    # top_indices: (current_batch_size, k)
                    # Result: (current_batch_size, k)
                    current_candidates = torch.gather(current_candidates, 1, top_indices)
                    stage_result['top_candidates'] = current_candidates.cpu().tolist()  # Store per-image candidates
                    stage_result['final_prediction'] = False

                stage_results.append(stage_result)

            # Set final variables for compatibility with existing code
            pred_labels = final_pred_labels
            batch_log_likelihood_tensor = stage_results[-1]['log_likelihood']  # Use last stage results for JSON saving

        # Update accuracy counters for the batch
        # Initialize correct_dict if this is the first batch
        if not correct_dict and top_k_data is not None:
            correct_dict = {k: 0 for k in top_k_data['valid_k_list']}

        # Compute top-k accuracy for all valid k values
        if top_k_data is not None:
            batch_correct_dict = {k: 0 for k in top_k_data['valid_k_list']}

            for k in top_k_data['valid_k_list']:
                _, top_k_indices = torch.topk(top_k_data['stage_log_likelihood_tensor'], k=k, dim=1)
                top_k_candidates = torch.gather(top_k_data['current_candidates'], 1, top_k_indices)

                if args.dataset == "objectnet":
                    # Map top-k candidates to ObjectNet space and check
                    for batch_idx in range(current_batch_size):
                        objectnet_top_k = [dataset_val.class_idx_map.get(c.item(), -1) for c in top_k_candidates[batch_idx]]
                        if label_batch[batch_idx].item() in objectnet_top_k:
                            batch_correct_dict[k] += 1
                else:
                    # Standard evaluation
                    is_correct = (top_k_candidates == label_batch.unsqueeze(1)).any(dim=1)
                    batch_correct_dict[k] += is_correct.sum().item()

            # Update global counters
            for k in top_k_data['valid_k_list']:
                correct_dict[k] += batch_correct_dict[k]

        # Maintain backward compatibility with top-1 accuracy
        if args.dataset == "objectnet":
            # For ObjectNet: map ImageNet predictions to ObjectNet class space
            # pred_labels contains ImageNet class indices, label_batch contains ObjectNet class indices
            objectnet_pred_labels = torch.tensor([dataset_val.class_idx_map.get(pred.item(), -1) for pred in pred_labels], device=device)
            batch_correct = (objectnet_pred_labels == label_batch).sum().item()
        else:
            # Standard evaluation for other datasets
            batch_correct = (pred_labels == label_batch).sum().item()
        correct += batch_correct
        total += current_batch_size
        
        # Save JSON files only if requested (this requires iteration over individual images)
        if args.save_json:
            for img_idx in range(current_batch_size):
                label = label_batch[img_idx]
                pred = pred_labels[img_idx]
                
                # For ObjectNet: map ImageNet prediction to ObjectNet class space for JSON saving
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
                    "multi_stage": num_stages > 1,
                    "num_stages": num_stages
                }

                # Add stage configuration
                data["stage_configs"] = {
                    "num_candidate_list": num_candidate_list,
                    "num_sample_list": num_sample_list,
                    "num_scale_list": num_scale_list
                }

                # Add detailed stage-by-stage results
                stages_data = []
                for stage_idx, stage_result in enumerate(stage_results):
                    stage_data = {
                        "stage": stage_result['stage'],
                        "num_candidates": stage_result['num_candidates'],
                        "num_samples": stage_result['num_samples'],
                        "num_scales": stage_result['num_scales'],
                        "sequence_length": stage_result['sequence_length'],
                        "log_likelihood": stage_result['log_likelihood'][img_idx].tolist(),  # Convert tensor to list
                        "candidates": stage_result['candidates'][img_idx],  # This image's specific candidates
                        "processing_time": stage_result['processing_time']
                    }

                    if 'top_candidates' in stage_result:
                        stage_data["top_candidates"] = stage_result['top_candidates'][img_idx]  # This image's top candidates

                    if stage_result.get('final_prediction', False):
                        stage_data["final_prediction"] = pred_for_json

                    stages_data.append(stage_data)

                data["stages"] = stages_data

                # Legacy compatibility fields
                if num_stages == 1:
                    # Single-stage compatibility
                    data["num_scale"] = num_scale_list[0]
                    data["sequence_length"] = stage_results[0]['sequence_length']
                    data["explanation"] = "Single-stage classification using log_likelihood score function with sum aggregation."
                else:
                    # Multi-stage explanation
                    data["explanation"] = f"Multi-stage classification with {num_stages} stages. Each stage filters candidates progressively."

                # Add sample augmentation metadata to JSON if used
                if max(num_sample_list) > 1:
                    data["max_samples"] = max(num_sample_list)
                    data["sigma"] = args.sigma
                    stages_with_augmentation = [i for i, s in enumerate(num_sample_list) if s > 1]
                    data["sample_augmentation_applied_to_stages"] = stages_with_augmentation

                # Add top-k accuracy metadata if available
                if top_k_data is not None:
                    data["top_k_metadata"] = {
                        "requested_k_list": k_list,
                        "valid_k_list": top_k_data['valid_k_list'],
                        "max_candidates": top_k_data['current_candidates'].shape[1]
                    }

                # Ensure all data is JSON serializable
                json_safe_data = make_json_serializable(data)

                with open(json_fname, "w") as f:
                    json.dump(json_safe_data, f, indent=4)

    # End timing and log runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    logging.info(f"Classification completed in {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    logging.info(f"Average time per sample: {total_runtime/total:.3f} seconds")
    
    # Add detailed timing breakdown for multi-stage
    if num_stages > 1:
        logging.info(f"Stage-specific timing breakdown:")
        for stage_idx in range(num_stages):
            stage_time = stage_total_times[stage_idx]
            stage_percentage = stage_time / total_runtime * 100
            logging.info(f"  Stage {stage_idx}: {stage_time:.2f} seconds ({stage_percentage:.1f}%)")
            logging.info(f"    Average Stage {stage_idx} time per sample: {stage_time/total:.3f} seconds")

    metric_name = "Scores (log_likelihood)"

    logging.info(f"\nOverall Accuracies using {metric_name} for Classification:")

    # Build accuracy description based on configuration
    accuracy_description = f"Overall Accuracy (d{args.depth}"

    if num_stages > 1:
        accuracy_description += f", multi-stage: {num_stages} stages"
        accuracy_description += f", candidates={num_candidate_list}"
        accuracy_description += f", scales={num_scale_list}"
    else:
        accuracy_description += f", single-stage: scale={num_scale_list[0]}"

    if max(num_sample_list) > 1:
        accuracy_description += f", sample_augmentation=max{max(num_sample_list)}"

    accuracy_description += f"): {100 * correct / total:.2f}%"
    logging.info(accuracy_description)

    # Report top-k accuracies if available
    if correct_dict:
        logging.info(f"\nTop-k Accuracies:")
        top_k_results = []
        for k in sorted(correct_dict.keys()):
            accuracy = 100 * correct_dict[k] / total
            top_k_results.append(f"Top-{k}: {accuracy:.2f}%")
        logging.info("  " + ", ".join(top_k_results))

    # Add sample augmentation summary
    if max(num_sample_list) > 1:
        logging.info(f"\nSample Augmentation Summary:")
        logging.info(f"  Maximum samples per image: {max(num_sample_list)}")
        logging.info(f"  Gaussian noise variance: {args.sigma}")
        stages_with_augmentation = [i for i, s in enumerate(num_sample_list) if s > 1]
        if stages_with_augmentation:
            logging.info(f"  Applied to stages: {stages_with_augmentation}")
        logging.info(f"  Total images processed: {total}")


if __name__ == "__main__":
    main()