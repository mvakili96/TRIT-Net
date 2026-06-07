import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ptsemseg.models.common import MyDecoder
from ptsemseg.models.common import nonlinearity


class Dblock_more_dilate(nn.Module):
    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out

        return out


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x


class DinkNet34(nn.Module):
    def __init__(self, n_classes_seg=19):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.n_classes_seg = n_classes_seg

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity

        self.finalconvSeg = nn.Conv2d(32, n_classes_seg, 3, padding=1)
        self.finalreluSeg = nonlinearity

        self.finalconvCent = nn.Conv2d(32 + n_classes_seg, 1, 1, stride=1, padding=0, bias=True)

        if self.n_classes_seg == 4:
            self.finalconvAFM = MyDecoder(in_channels=32 + n_classes_seg, out_channels=1)

    def forward(self, x):
        size_in = x.size()
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.dblock(e4)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2

        decoder_out_this = self.decoder2(d3)
        if e1.shape[2] != decoder_out_this.shape[2] or e1.shape[3] != decoder_out_this.shape[3]:
            decoder_out_this = F.interpolate(
                decoder_out_this,
                size=(e1.shape[2], e1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
            d2 = decoder_out_this + e1

        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)

        backbone = self.finalrelu2(out)

        outseg = self.finalconvSeg(backbone)
        outsegrelu = self.finalreluSeg(outseg)

        backbone_dlinknet = torch.cat([backbone, outsegrelu], 1)

        outcent_final = self.finalconvCent(backbone_dlinknet)
        outcent_final = F.interpolate(
            outcent_final, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
        )

        if self.n_classes_seg == 4:
            out_AFM = self.finalconvAFM(backbone_dlinknet)
            out_AFM_final = F.interpolate(
                out_AFM, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
            )

        out_seg_final = F.interpolate(
            outseg, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
        )

        if self.n_classes_seg == 4:
            return out_seg_final, outcent_final, out_AFM_final
        elif self.n_classes_seg == 3:
            return out_seg_final, outcent_final
