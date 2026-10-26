"""Locate the separately supplied original hardware project without vendoring it."""
import os
from pathlib import Path

# Existing workstation fallback preserves historical command compatibility.
HARDWARE_ROOT = Path(os.environ.get('RBF_HARDWARE_ROOT', '/home/ziyan/SURF2026/rbf_hardware')).expanduser().resolve()
RESPONSE_BANK = HARDWARE_ROOT / 'dataset/empirical/hardware_response_samples.npz'
