"""Helpers for saving and loading per-distribution ("task") signatures.

Each detected distribution stores: the ridge accumulators (``global_At_A``,
``global_AtA_w``), the closed-form IA3 update derived from them, and the
Gaussian-mixture task key (means/vars/counts). These were previously saved
and re-loaded with near-identical, copy-pasted code in main.py and
evaluate.py.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


def task_checkpoint_dir(output_dir: str, task_id: int) -> str:
    """Path to the checkpoint directory for a given detected distribution."""
    return os.path.join(output_dir, f"ia3_task_{task_id}")


def save_task_checkpoint(output_dir, task_id, global_At_A, global_AtA_w,
                          global_updates=None, head_weight=None, head_bias=None):
    """Persist the ridge accumulators and the GMM task key for ``task_id``."""
    ckpt_dir = task_checkpoint_dir(output_dir, task_id)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    torch.save(global_At_A, os.path.join(ckpt_dir, "global_At_A.pth"))
    torch.save(global_AtA_w, os.path.join(ckpt_dir, "global_AtA_w.pth"))

    if global_updates is not None:
        torch.save(global_updates, os.path.join(ckpt_dir, "global_updates.pth"))
    if head_weight is not None:
        torch.save(head_weight, os.path.join(ckpt_dir, "head_weight.pth"))
    if head_bias is not None:
        torch.save(head_bias, os.path.join(ckpt_dir, "head_bias.pth"))

    return ckpt_dir

def load_task_signature(output_dir, task_id):
    """Load the masked GMM task key and IA3 ridge solution for ``task_id``."""
    ckpt_dir = task_checkpoint_dir(output_dir, task_id)
    global_updates = torch.load(os.path.join(ckpt_dir, "global_updates.pth"))
    return { "global_updates": global_updates}
