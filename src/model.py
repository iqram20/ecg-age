import torch
import torch.nn as nn


class LeadAttention(nn.Module):
    def __init__(self, in_ch: int = 24):
        super().__init__()

        hidden = max(8, in_ch // 4)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_ch, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, in_ch, kernel_size=1),
        )

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=1)
        return x * weights


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()

        c1 = out_ch // 3
        c2 = out_ch // 3
        c3 = out_ch - c1 - c2

        self.b1 = nn.Conv1d(
            in_ch, c1, kernel_size=3,
            stride=stride, padding=1
        )
        self.b2 = nn.Conv1d(
            in_ch, c2, kernel_size=7,
            stride=stride, padding=3
        )
        self.b3 = nn.Conv1d(
            in_ch, c3, kernel_size=15,
            stride=stride, padding=7
        )

        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = torch.cat(
            [self.b1(x), self.b2(x), self.b3(x)],
            dim=1,
        )
        return self.act(self.bn(x))


class CNNStem(nn.Module):
    def __init__(self, in_ch: int = 24, d_model: int = 192):
        super().__init__()

        self.net = nn.Sequential(
            MultiScaleConvBlock(in_ch, 64, stride=2),
            MultiScaleConvBlock(64, 128, stride=2),
            MultiScaleConvBlock(128, d_model, stride=2),
        )

    def forward(self, x):
        return self.net(x)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.bn(self.conv(x)))
        return x + residual


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 192, nhead: int = 4):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )

        self.tx = nn.TransformerEncoder(
            layer,
            num_layers=1,
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tx(x)
        return x.transpose(1, 2)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int = 192):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Conv1d(
                d_model,
                d_model // 2,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                d_model // 2,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        weights = torch.softmax(
            self.attn(x),
            dim=2,
        )
        return (x * weights).sum(dim=2)


class ECGAgeRegressor(nn.Module):
    """
    Morphology-aware CNN-TCN-Transformer ECG-age regressor.

    Frozen manuscript configuration:
        input channels : 24
        d_model        : 192
        transformer depth : 6
        attention heads   : 4
    """

    def __init__(
        self,
        in_ch: int = 24,
        d_model: int = 192,
        depth: int = 6,
        nhead: int = 4,
    ):
        super().__init__()

        self.lead_attention = LeadAttention(in_ch)
        self.stem = CNNStem(in_ch, d_model)

        self.tcn = nn.Sequential(
            TCNBlock(d_model, dilation=1),
            TCNBlock(d_model, dilation=2),
            TCNBlock(d_model, dilation=4),
            TCNBlock(d_model, dilation=8),
            TCNBlock(d_model, dilation=16),
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                )
                for _ in range(depth)
            ]
        )

        self.att_pool = AttentionPooling(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = self.lead_attention(x)
        x = self.stem(x)
        x = self.tcn(x)

        for block in self.blocks:
            x = block(x)

        att_feat = self.att_pool(x)
        mean_feat = x.mean(dim=2)

        pooled = (
            0.5 * att_feat
            + 0.5 * mean_feat
        )

        return self.head(pooled).squeeze(1)


def count_trainable_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
