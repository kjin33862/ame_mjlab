"""Device selection for AME component tests."""

from __future__ import annotations

import os

import torch


def get_test_device() -> str:
  """Use CUDA when available unless FORCE_CPU requests CPU execution."""
  return (
    "cuda" if torch.cuda.is_available() and os.getenv("FORCE_CPU") != "1" else "cpu"
  )


__all__ = ["get_test_device"]
