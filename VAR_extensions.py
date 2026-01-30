"""
VAR Extensions - Custom modifications to the VAR submodule

This module monkey-patches the VAR submodule to add custom functions needed
for classification without modifying the upstream VAR repository.

Custom functions added:
- VQVAE.img_to_post: Extract post-quantization features for classification
"""

import sys
import os.path as osp
from typing import Optional, Sequence, Union, Tuple

# Add VAR directory to Python path
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), 'VAR'))

# Import original classes
from models.vqvae import VQVAE
from models import build_vae_var

# ============================================================================
# Custom VQVAE Extensions
# ============================================================================

def img_to_post(self, inp_img_no_grad, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None):
    """
    Extract post-encoder, post-quant_conv features from input image.

    This function is used in classification to get the latent features
    before quantization, which are then quantized for each candidate class.

    Args:
        inp_img_no_grad: Input image tensor (no grad required)
        v_patch_nums: Optional patch numbers (unused in this function)

    Returns:
        f: Post-quant_conv features
    """
    f = self.quant_conv(self.encoder(inp_img_no_grad))
    return f

# Monkey-patch the function into VQVAE
VQVAE.img_to_post = img_to_post

# ============================================================================
# Re-export with extensions applied
# ============================================================================

__all__ = ['VQVAE', 'build_vae_var']
