import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

from ptsemseg.models.common import MyDecoder
from ptsemseg.models.common import backbone_url


class ConvBNReLU(nn.Module):
    def __init__(
        self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, groups=1, bias=False
    ):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class StemBlock(nn.Module):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat


class CEBlock(nn.Module):
    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class GELayerS1(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan,
                mid_chan,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_chan,
                bias=False,
            ),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan,
                mid_chan,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_chan,
                bias=False,
            ),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan,
                mid_chan,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=mid_chan,
                bias=False,
            ),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_chan, in_chan, kernel_size=3, stride=2, padding=1, groups=in_chan, bias=False
            ),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)

        return feat


class DetailBranch(nn.Module):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2, padding="valid"),
            ConvBNReLU(128, 128, 3, stride=1, padding="same"),
            ConvBNReLU(128, 128, 3, stride=1, padding="same"),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)

        return feat


class SegmentBranch(nn.Module):
    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)

        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):
    def __init__(self):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False),
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)

        size_in = left1.size()
        right1 = F.relu(
            F.interpolate(right1, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)
        )

        left = left1 * right1
        right = left2 * F.relu(right2)

        right = F.interpolate(right, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)
        out = self.conv(left + right)

        return out


class SegmentHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=False):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(nn.Upsample(scale_factor=2), ConvBNReLU(mid_chan, mid_chan2, 3, stride=1))
            if aux
            else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=1, mode="bilinear", align_corners=False),
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)

        return feat


class SegmentationHead(nn.Module):
    def __init__(self, in_chan, n_classes):
        super(SegmentationHead, self).__init__()

        self.conv_out = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(in_chan, n_classes, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=1, mode="bilinear", align_corners=False),
        )

    def forward(self, x):
        feat = self.conv_out(x)

        return feat


class Bisenet_v2(nn.Module):
    def __init__(self, n_classes_seg=19):
        super().__init__()
        self.n_classes_seg = n_classes_seg

        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()
        self.head = SegmentHead(128, 1024, n_classes_seg, up_factor=8)
        self.aux2 = SegmentHead(16, 128, n_classes_seg, up_factor=4)
        self.aux3 = SegmentHead(32, 128, n_classes_seg, up_factor=8)
        self.aux4 = SegmentHead(64, 128, n_classes_seg, up_factor=16)
        self.aux5_4 = SegmentHead(128, 128, n_classes_seg, up_factor=32)

        self.finalconvCent = nn.Conv2d(128 + n_classes_seg, 1, 1, stride=1, padding=0, bias=True)

        if n_classes_seg == 4:
            self.finalconvAFM = MyDecoder(in_channels=128 + n_classes_seg, out_channels=1)

    def forward(self, x):
        size_in = x.size()
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)

        outseg = self.head(feat_head)

        logits_aux2 = self.aux2(feat2)
        logits_aux3 = self.aux3(feat3)
        logits_aux4 = self.aux4(feat4)
        logits_aux5_4 = self.aux5_4(feat5_4)

        outseg_relu = F.relu(outseg)
        backbone = torch.cat([feat_head, outseg_relu], 1)

        out_centerline = self.finalconvCent(backbone)

        if self.n_classes_seg == 4:
            out_AFM = self.finalconvAFM(backbone)
            out_AFM_final = F.interpolate(
                out_AFM, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
            )

        outseg_final = F.interpolate(
            outseg, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
        )

        out_aux1 = F.interpolate(
            logits_aux2, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
        )
        out_aux2 = F.interpolate(
            logits_aux3, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
        )
        out_aux3 = F.interpolate(
            logits_aux4, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
        )
        out_aux4 = F.interpolate(
            logits_aux5_4, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
        )

        out_centerline_final = F.interpolate(
            out_centerline, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True
        )

        if self.n_classes_seg == 4:
            return (
                outseg_final,
                out_centerline_final,
                out_AFM_final,
                out_aux1,
                out_aux2,
                out_aux3,
                out_aux4,
            )
        elif self.n_classes_seg == 3:
            return outseg_final, out_centerline_final, out_aux1, out_aux2, out_aux3, out_aux4

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if not module.bias is None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, "last_bn") and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def load_pretrain(self):
        state = modelzoo.load_url(backbone_url)
        for name, child in self.named_children():
            if name in state.keys():
                child.load_state_dict(state[name], strict=True)
