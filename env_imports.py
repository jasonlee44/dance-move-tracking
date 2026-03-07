"""
Standard imports for PyTorch deep learning projects.
Use this as a reference or copy the blocks you need into your modules.
"""

# -----------------------------------------------------------------------------
# PyTorch
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Optional: vision / audio
# import torchvision
# import torchvision.transforms as T
# import torchaudio

# -----------------------------------------------------------------------------
# Device selection (CPU, CUDA, or MPS on Apple Silicon)
# -----------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

# -----------------------------------------------------------------------------
# Numeric & scientific
# -----------------------------------------------------------------------------
import numpy as np

# -----------------------------------------------------------------------------
# Visualization & logging
# -----------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------
# Example: move model and batch to device
# -----------------------------------------------------------------------------
# model = MyModel().to(DEVICE)
# inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
