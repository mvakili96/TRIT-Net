from functools import partial

import torch.nn as nn
import torch.nn.functional as F


nonlinearity = partial(F.relu, inplace=False)

backbone_url = "https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth"


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        dropout=0.1,
        dilation_this=1,
    ):
        super().__init__()
        if dilation_this == 1:
            self.add_module(
                "conv",
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=kernel // 2,
                    bias=False,
                    dilation=dilation_this,
                ),
            )
            self.add_module("norm", nn.BatchNorm2d(out_channels))
            self.add_module("relu", nn.ReLU(inplace=True))
        else:
            self.add_module(
                "conv",
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    padding="same",
                    bias=False,
                    dilation=dilation_this,
                ),
            )
            self.add_module("norm", nn.BatchNorm2d(out_channels))
            self.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        return super().forward(x)


class MyDecoder(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
        )
        self.add_module("norm", nn.BatchNorm2d(16))
        self.add_module("relu", nn.ReLU(inplace=True))

        self.add_module(
            "conv_b",
            nn.Conv2d(
                in_channels=16,
                out_channels=48,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True,
            ),
        )
        self.add_module("norm_b", nn.BatchNorm2d(48))
        self.add_module("relu_b", nn.ReLU(inplace=True))

        self.add_module(
            "conv_c",
            nn.Conv2d(
                in_channels=48,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
        )
