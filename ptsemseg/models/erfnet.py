import torch
import torch.nn as nn
import torch.nn.functional as F

from ptsemseg.models.common import MyDecoder


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        s1 = 0
        s2 = 0
        if input.shape[2] % 2 == 1:
            s1 = 1
        elif input.shape[3] % 2 == 1:
            s2 = 1

        self.pool.padding = (s1, s2)
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(
            chann,
            chann,
            (3, 1),
            stride=1,
            padding=(1 * dilated, 0),
            bias=True,
            dilation=(dilated, 1),
        )

        self.conv1x3_2 = nn.Conv2d(
            chann,
            chann,
            (1, 3),
            stride=1,
            padding=(0, 1 * dilated),
            bias=True,
            dilation=(1, dilated),
        )

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)
        return F.relu(output + input)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for _ in range(0, 5):
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for _ in range(0, 2):
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.finalconvSeg = nn.ConvTranspose2d(
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True
        )

        self.finalconvCent = nn.Conv2d(16 + num_classes, 1, 1, stride=1, padding=0, bias=True)

        if num_classes == 4:
            self.finalconvAFM = MyDecoder(in_channels=16 + num_classes, out_channels=1)

    def forward(self, input, size_in):
        output = input

        for layer in self.layers:
            output = layer(output)

        backbone = output
        outseg = self.finalconvSeg(backbone)
        outsegrelu = F.relu(outseg)

        backbone = F.interpolate(
            backbone, size=(outseg.size()[2], outseg.size()[3]), mode="bilinear", align_corners=True
        )
        backbone_erfnet = torch.cat([backbone, outsegrelu], 1)

        outcent_final = self.finalconvCent(backbone_erfnet)
        outcent_final = F.interpolate(
            outcent_final, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
        )

        if self.num_classes == 4:
            out_AFM = self.finalconvAFM(backbone_erfnet)
            out_AFM_final = F.interpolate(
                out_AFM, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
            )

        outseg_final = F.interpolate(
            outseg, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
        )

        if self.num_classes == 4:
            return outseg_final, outcent_final, out_AFM_final
        elif self.num_classes == 3:
            return outseg_final, outcent_final


class ERFNet(nn.Module):
    def __init__(self, n_classes_seg=19, n_channels_reg=3):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(n_classes_seg)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            size = input.size()
            output = self.encoder(input)
            return self.decoder.forward(output, size)
