"""
benchmarks/models/cnn.py
========================
4-layer spatial CNN for multi-source rainfall field fusion.

Reference
---------
Wu et al. (2020) "A spatiotemporal deep fusion model for merging satellite
and gauge precipitation in China", Journal of Hydrology 584:124664.

The paper compares CNN, LSTM, and CNN-LSTM variants; this module implements
the standalone CNN baseline (Section 3.3 of the paper) adapted for three
input channels: IDW-interpolated rain gauge grid, IDW-interpolated CML
specific-attenuation grid, and weather radar reflectivity grid.
"""

import torch
import torch.nn as nn


class RainfallCNN(nn.Module):
    """Spatial CNN for fusing multi-source rainfall observations.

    Input  : [B, in_channels, H, W]  — stacked gridded observation channels
    Output : [B, 1, H, W]            — predicted rainfall field
                                       No output activation; clamp >= 0 at
                                       evaluation time (same convention as GNN).
    """

    def __init__(self, in_channels: int = 3, hidden: int = 64):
        super().__init__()
        self.config = dict(in_channels=in_channels, hidden=hidden)

        self.net = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Conv2d(32, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            # Output projection (1×1 conv = per-pixel linear)
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
