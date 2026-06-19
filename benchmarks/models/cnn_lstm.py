"""
benchmarks/models/cnn_lstm.py
=============================
LSTM and CNN-LSTM rainfall-fusion models with channel-wise input normalisation.

Following Wu et al. (2020) "A spatiotemporal deep fusion model for merging
satellite and gauge precipitation in China", Journal of Hydrology 584:124664.

All three models share a unified interface:
  Input  : [B, S, C, H, W]  — batch, sequence length, channels, height, width
  Output : [B, 1, H, W]     — predicted rainfall field

Additions over RainfallCNN (run_cnn.py)
-----------------------------------------
1. ChannelNorm  – channel-wise z-score normalisation with statistics fitted
                  on training data and saved in the model checkpoint.
2. RainfallCNNNorm  – same 4-layer CNN but with ChannelNorm prepended.
                      Use with seq_len=1; the extra S dimension is squeezed.
3. RainfallLSTM     – per-pixel LSTM applied directly to the raw input
                      channels; no CNN spatial encoder.
4. RainfallCNNLSTM  – 2-layer CNN spatial encoder (shared weights across
                      timesteps) followed by a per-pixel LSTM that aggregates
                      temporal context.  Closest to Wu et al.'s main model.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Channel-wise z-score normalisation
# ---------------------------------------------------------------------------

class ChannelNorm(nn.Module):
    """
    Channel-wise z-score normalisation with statistics stored as buffers.

    Buffers (mean, std) are serialised with the model checkpoint so that
    inference uses the exact same statistics as training without needing
    to pass them separately.

    Accepts both single-frame [B, C, H, W] and sequence [B, S, C, H, W]
    inputs by treating the third-from-last axis as the channel dimension.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_channels))
        self.register_buffer("std",  torch.ones(num_channels))

    def fit(self, data: torch.Tensor) -> None:
        """
        Compute per-channel statistics from training data.

        Parameters
        ----------
        data : torch.Tensor  shape [T, C, H, W]
            All training input grids (before sequencing).
        """
        self.mean = data.mean(dim=(0, 2, 3)).to(self.mean.device)
        self.std  = data.std( dim=(0, 2, 3)).clamp(min=1e-6).to(self.std.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Broadcast over both 4-D [B, C, H, W] and 5-D [B, S, C, H, W] inputs.
        # dim -3 is the channel axis in both cases.
        shape = [1] * x.ndim
        shape[-3] = -1
        return (x - self.mean.view(shape)) / self.std.view(shape)


# ---------------------------------------------------------------------------
# CNN + normalisation  (single-frame; seq_len=1)
# ---------------------------------------------------------------------------

class RainfallCNNNorm(nn.Module):
    """
    4-layer spatial CNN with channel-wise z-score input normalisation.

    Architecture identical to RainfallCNN (run_cnn.py) with ChannelNorm
    prepended.  The extra sequence dimension (S=1) is accepted and squeezed
    so the same SequenceGridDataset used for LSTM/CNN-LSTM can drive this
    model without code changes.

    Input  : [B, 1, C, H, W]  or  [B, C, H, W]
    Output : [B, 1, H, W]
    """

    def __init__(self, in_channels: int = 3, hidden: int = 64):
        super().__init__()
        self.config = dict(model_type="cnn", in_channels=in_channels,
                           hidden=hidden)
        self.norm = ChannelNorm(in_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32,     kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,         hidden,  kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden,     hidden,  kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            x = x[:, -1]        # [B, 1, C, H, W] → [B, C, H, W]
        return self.net(self.norm(x))


# ---------------------------------------------------------------------------
# LSTM  (per-pixel; no CNN encoder)
# ---------------------------------------------------------------------------

class RainfallLSTM(nn.Module):
    """
    Per-pixel LSTM for temporal fusion of multi-source gridded rainfall inputs.

    The input sequence [B, S, C, H, W] is reshaped to [B·H·W, S, C] so that
    each spatial location is processed as an independent sequence by a shared
    LSTM.  The final hidden state at each location is projected to a single
    rainfall value, producing a full [B, 1, H, W] output field.

    Parameters
    ----------
    in_channels : int   number of input channels (default 3)
    hidden      : int   LSTM hidden-state size (default 64)
    num_layers  : int   number of stacked LSTM layers (default 1)
    """

    def __init__(self, in_channels: int = 3, hidden: int = 64,
                 num_layers: int = 1):
        super().__init__()
        self.config = dict(model_type="lstm", in_channels=in_channels,
                           hidden=hidden, num_layers=num_layers)
        self.norm = ChannelNorm(in_channels)
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C, H, W]
        B, S, C, H, W = x.shape
        x = self.norm(x)
        x = x.permute(0, 3, 4, 1, 2)                       # [B, H, W, S, C]
        x = x.reshape(B * H * W, S, C)                     # [B·H·W, S, C]
        _, (h_n, _) = self.lstm(x)                         # h_n: [layers, B·H·W, hidden]
        feat = h_n[-1]                                      # [B·H·W, hidden]
        feat = feat.view(B, H, W, -1).permute(0, 3, 1, 2) # [B, hidden, H, W]
        return self.out(feat)                               # [B, 1, H, W]


# ---------------------------------------------------------------------------
# CNN-LSTM  (spatial encoder + per-pixel temporal aggregation)
# ---------------------------------------------------------------------------

class RainfallCNNLSTM(nn.Module):
    """
    CNN spatial encoder + per-pixel LSTM for spatiotemporal fusion.

    Closest adaptation of the Wu et al. (2020) CNN-LSTM design to our data:

    1. A 2-layer CNN encoder (shared weights) extracts spatial feature maps
       [cnn_hidden, H, W] from each frame in the input sequence.
    2. For each spatial location, the sequence of CNN features is passed
       through a shared LSTM that aggregates temporal context.
    3. The LSTM's final hidden state at each location is projected to a
       single rainfall value via a 1×1 convolution.

    Parameters
    ----------
    in_channels : int   number of input channels (default 3)
    cnn_hidden  : int   CNN feature-map channels after spatial encoding (default 32)
    lstm_hidden : int   LSTM hidden-state size (default 64)
    num_layers  : int   number of stacked LSTM layers (default 1)
    """

    def __init__(self, in_channels: int = 3, cnn_hidden: int = 32,
                 lstm_hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.config = dict(model_type="cnn_lstm", in_channels=in_channels,
                           cnn_hidden=cnn_hidden, lstm_hidden=lstm_hidden,
                           num_layers=num_layers)
        self.norm = ChannelNorm(in_channels)

        # Spatial encoder — shared across all timesteps in the sequence
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32,        kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,         cnn_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_hidden),
            nn.ReLU(inplace=True),
        )

        # Per-pixel temporal aggregation
        self.lstm = nn.LSTM(
            input_size=cnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
        )

        self.out = nn.Conv2d(lstm_hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C, H, W]
        B, S, C, H, W = x.shape
        x = self.norm(x)

        # CNN: encode all S frames in a single batched call
        x    = x.view(B * S, C, H, W)                      # [B·S, C, H, W]
        feat = self.cnn(x)                                  # [B·S, cnn_hidden, H, W]
        feat = feat.view(B, S, -1, H, W)                   # [B, S, cnn_hidden, H, W]

        # LSTM: per-pixel temporal aggregation
        feat = feat.permute(0, 3, 4, 1, 2)                 # [B, H, W, S, cnn_hidden]
        feat = feat.reshape(B * H * W, S, -1)              # [B·H·W, S, cnn_hidden]
        _, (h_n, _) = self.lstm(feat)                      # h_n: [layers, B·H·W, lstm_h]
        out  = h_n[-1]                                      # [B·H·W, lstm_hidden]
        out  = out.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, lstm_hidden, H, W]
        return self.out(out)                                # [B, 1, H, W]
