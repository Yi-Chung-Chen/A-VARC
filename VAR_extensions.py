"""
VAR Extensions - Custom modifications to the VAR submodule

This module monkey-patches the VAR submodule to add custom functions needed
for classification without modifying the upstream VAR repository.

Custom functions added:
- VQVAE.img_to_post: Extract post-quantization features for classification
- VAR.forward: Fix ed computation to support partial-scale inference
"""

import sys
import os.path as osp
from typing import Optional, Sequence, Union, Tuple

import torch

# Add VAR directory to Python path
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), 'VAR'))

# Import original classes
from models.vqvae import VQVAE
from models.var import VAR
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
# Custom VAR Extensions
# ============================================================================

def var_forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:
    """
    Patched VAR.forward that derives sequence end (ed) from the actual input
    length instead of always using self.L.  This allows partial-scale inference
    (e.g. num_scale_list < 10) without a shape mismatch on the positional
    embeddings.

    The only change vs. the original is:
        ed = self.L  →  ed = x_BLCv_wo_first_l.shape[1] + self.first_l
    """
    bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, x_BLCv_wo_first_l.shape[1] + self.first_l)
    B = x_BLCv_wo_first_l.shape[0]
    with torch.cuda.amp.autocast(enabled=False):
        label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
        sos = cond_BD = self.class_emb(label_B)
        sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

        if self.prog_si == 0: x_BLC = sos
        else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
        x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]

    attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
    cond_BD_or_gss = self.shared_ada_lin(cond_BD)

    temp = x_BLC.new_ones(8, 8)
    main_type = torch.matmul(temp, temp).dtype

    x_BLC = x_BLC.to(dtype=main_type)
    cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
    attn_bias = attn_bias.to(dtype=main_type)

    for b in self.blocks:
        x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
    x_BLC = self.get_logits(x_BLC.float(), cond_BD)

    if self.prog_si == 0:
        if isinstance(self.word_embed, torch.nn.Linear):
            x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
        else:
            s = 0
            for p in self.word_embed.parameters():
                if p.requires_grad:
                    s += p.view(-1)[0] * 0
            x_BLC[0, 0, 0] += s
    return x_BLC

# Monkey-patch VAR.forward to support partial-scale inference
VAR.forward = var_forward

# ============================================================================
# Re-export with extensions applied
# ============================================================================

__all__ = ['VQVAE', 'VAR', 'build_vae_var']
