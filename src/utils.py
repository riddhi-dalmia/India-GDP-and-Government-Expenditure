"""
utils.py — Shared constants and helper functions
=================================================
This file holds things that multiple modules need, so we don't repeat ourselves.
Think of it as the "toolbox" for the whole project.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW   = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC  = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR    = os.path.join(BASE_DIR, "outputs", "figures")
RES_DIR    = os.path.join(BASE_DIR, "outputs", "results")

# ── Colour palette used across all charts ──────────────────────────────────────
COLOR_HIGH = "#2196F3"   # blue  → high-growth years
COLOR_LOW  = "#F44336"   # red   → low-growth years
COLOR_ALL  = "#4CAF50"   # green → full-period

# ── Threshold logic ────────────────────────────────────────────────────────────
# We will compute this dynamically from the data (median split),
# but keep the label strings here so every module uses the same wording.
LABEL_HIGH = "High-Growth Period"
LABEL_LOW  = "Low-Growth Period"

def ensure_dirs():
    """Create output folders if they don't exist yet."""
    for d in [DATA_RAW, DATA_PROC, FIG_DIR, RES_DIR]:
        os.makedirs(d, exist_ok=True)
