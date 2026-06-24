# 2020/7/21
# Jungwon Kang

# Note that the term 'TPEnet' and 'rpnet' are mixed, but both terms are used to indicate 'TPEnet'.
import copy

import cv2
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptsemseg.evaluation import visualize_featuremap

from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu, inplace=True)

import numpy as np
from scipy.signal import find_peaks

########################################################################################################################
class ConvLayer(nn.Sequential):
    """ConvLayer - CBR module, where CBR: Conv + BN + ReLU"""

    ###======================================================================================================
    ### ConvLayer::__init__()
    ###======================================================================================================
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel//2, bias = False))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
    #end


    ###======================================================================================================
    ### ConvLayer::forward()
    ###======================================================================================================
    def forward(self, x):
        return super().forward(x)
    #end
#end



class MyDecoder(nn.Sequential):
    ###======================================================================================================
    ### MyDecoder::__init__()
    ###======================================================================================================
    def __init__(self, in_channels,out_channels):
        super().__init__()
        ###
        self.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=16,
                                          kernel_size=(1, 1), stride=1, padding=0, bias=True))
        self.add_module('norm', nn.BatchNorm2d(16))
        self.add_module('relu', nn.ReLU(inplace=True))

        ###
        self.add_module('conv_b', nn.Conv2d(in_channels=16, out_channels=48,
                                            kernel_size=(3, 3), stride=1, padding=0, bias=True))
        self.add_module('norm_b', nn.BatchNorm2d(48))
        self.add_module('relu_b', nn.ReLU(inplace=True))

        ###

        self.add_module('conv_c', nn.Conv2d(in_channels=48, out_channels=out_channels,
                                            kernel_size=(1, 1), stride=1, padding=0, bias=True))



    ###======================================================================================================
    ### MyDecoder::forward()
    ###======================================================================================================
    def forward(self, x):
        return super().forward(x)
    #end
#end


########################################################################################################################
class HarDBlock(nn.Module):
    """HarDBlock"""

    ###======================================================================================================
    ### HarDBlock::get_link()
    ###======================================================================================================
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        #end

        out_channels = growth_rate

        link = []

        for i in range(10):
            dv = 2 ** i

            if layer % dv == 0:
                k = layer - dv
                link.append(k)

                if i > 0:
                    out_channels *= grmul
                #end
            #end
        #end

        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0

        for i in link:
            ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        #end

        return out_channels, in_channels, link
    #end


    ###======================================================================================================
    ### HarDBlock::get_out_ch()
    ###======================================================================================================
    def get_out_ch(self):
        return self.out_channels
    #end


    ###======================================================================================================
    ### HarDBlock::__init__()
    ###======================================================================================================
    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False):
        super().__init__()

        ###
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0       # if upsample else in_channels


        ###
        for i in range(n_layers):
            outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out

            layers_.append( ConvLayer(inch, outch) )

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
            #end


            # print("idx_layers: [%d]" % i)
            # print(link)
            # print("inch: [%d], outch: [%d]" % (inch, outch))
        #end


        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)
    #end


    ###======================================================================================================
    ### HarDBlock::forward()
    ###======================================================================================================
    def forward(self, x):

        layers_ = [x]

        for layer in range(len(self.layers)):
            ###
            link = self.links[layer]
            tin = []


            ###
            for i in link:
                tin.append(layers_[i])
            #end


            ###
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            #end


            ###
            out = self.layers[layer](x)
            layers_.append(out)
        #end

        t = len(layers_)

        out_ = []

        for i in range(t):
            if (i == 0 and self.keepBase) or \
               (i == t-1) or (i%2 == 1):
                out_.append(layers_[i])
            #end
        #end

        out = torch.cat(out_, 1)
        return out
    #end


########################################################################################################################
class TransitionUp(nn.Module):
    """TransitionUp"""

    ###======================================================================================================
    ### TransitionUp::__init__()
    ###======================================================================================================
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #print("upsample",in_channels, out_channels)
    #end

    ###======================================================================================================
    ### TransitionUp::forward()
    ###======================================================================================================
    def forward(self, x, skip, concat=True):
        out = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode="bilinear",
                align_corners=True)

        if concat:                            
            out = torch.cat([out, skip], 1)
        #end
          
        return out
    #end
#end


########################################################################################################################
class TPEnet_a(nn.Module):
    """TPEnet"""

    ############################################################################################################
    ### TPEnet_a::__init__()
    ############################################################################################################
    def __init__(self, n_classes=3, n_channels_reg = 1):
        super(TPEnet_a, self).__init__()

        ###================================================================================================
        ### parameters (for FC-HarDNet)
        ###================================================================================================

        ### output ch of init-conv
        first_ch = [16, 24, 32, 48]

        ### output of [HDB-xD & TDx], where HDB-xD is HDB-xDown, TDx is Transition Down-x
        ch_list  = [64, 96, 160, 224, 320]

        ###
        grmul    = 1.7

        ### output of first bottleneck layer in each [HDB-xD & TDx]
        gr       = [10, 16, 18, 24, 32]

        ### the number of bottleneck layers in each [HDB-xD & TDx]
        n_layers = [4, 4, 8, 8, 8]

        ###
        blks = len(n_layers)            # => blks: 5


        ###================================================================================================
        ### parameters (for rpnet)
        ###================================================================================================
        ch_output_fc_hardnet_a = 48         # output of HDB-3U
        ch_output_fc_hardnet_b = n_classes         # output of Conv-Final

        self.n_classes = n_classes
        self.n_channels_reg = n_channels_reg
        ###================================================================================================
        ### init modules (for rpnet)
        ###================================================================================================
        #self.rpnet_decoder_cnvs = None
        self.rpnet_decoder_hmap_center = None


        ###================================================================================================
        ### init modules (for FC-HarDNet)
        ###================================================================================================
        self.base = nn.ModuleList([])


        ###================================================================================================
        ### set init-conv (for FC-HarDNet)
        ###================================================================================================
        self.base.append( ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2) )  # conv0
        self.base.append( ConvLayer(first_ch[0], first_ch[1],  kernel=3) )                          # conv1
        self.base.append( ConvLayer(first_ch[1], first_ch[2],  kernel=3, stride=2) )                # conv2
        self.base.append( ConvLayer(first_ch[2], first_ch[3],  kernel=3) )                          # conv3
            # created
            #   (conv0): Conv2d( 3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            #   (conv1): Conv2d(16, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            #   (conv2): Conv2d(24, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            #   (conv3): Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)


        ###================================================================================================
        ### down-network
        ###================================================================================================
        self.shortcut_layers           = []     # self.shortcut_layers          = [4, 7, 10, 13]
        self.set_idx_module_HarDBlock  = []     # self.set_idx_module_HarDBlock = [4, 7, 10, 13, 16]
        skip_connection_channel_counts = []

        ###
        ch = first_ch[3]                        # output ch of last conv in init-conv

        ###
        for i_blk in range(blks):
            ###///////////////////////////////////////////////////////////////////////////////////////
            ### <<FC-HarDNet - base>>
            ###///////////////////////////////////////////////////////////////////////////////////////

            ###------------------------------------------------------------------------------
            ### create HarDBlock
            ###------------------------------------------------------------------------------
            #print('idx_blk:[%d], ch_in:[%d]' % (i,ch))
            blk = HarDBlock(ch, gr[i_blk], grmul, n_layers[i_blk])
            ch = blk.get_out_ch()
            ch_out_HarDBlock_this = blk.get_out_ch()
            #print('idx_blk:[%d], ch_out:[%d]' % (i,ch))


            ###------------------------------------------------------------------------------
            ### append HarDBlock & create shortcut
            ###------------------------------------------------------------------------------

            ### append ch into skip_connection_channel_counts
            skip_connection_channel_counts.append(ch)

            ### append HarDBlock
            self.base.append( blk )                                               # APPEND

            ### append set_idx_module_HarDBlock
            self.set_idx_module_HarDBlock.append( len(self.base)-1 )              # APPEND

            ### append module index (in ModuleList) into self.shortcut_layers
            if i_blk < (blks-1):
                self.shortcut_layers.append( len(self.base)-1 )
            #end


            ###------------------------------------------------------------------------------
            ### create transition-down layer
            ###------------------------------------------------------------------------------
            #print('idx_blk:[%d], conv for ch_in:[%d]' % (i_blk,ch))
            self.base.append( ConvLayer(ch, ch_list[i_blk], kernel=1) )           # APPEND
            ch = ch_list[i_blk]
            #print('idx_blk:[%d], conv for ch_out:[%d]' % (i_blk,ch))


            ###------------------------------------------------------------------------------
            ### create AvgPool2d
            ###------------------------------------------------------------------------------
            if i_blk < blks-1:
                self.base.append( nn.AvgPool2d(kernel_size=2, stride=2) )         # APPEND
            #end


            ###///////////////////////////////////////////////////////////////////////////////////////
            ### <<RPNet - backbone>>
            ###///////////////////////////////////////////////////////////////////////////////////////
            # print('idx_blk:[%d], ch_out_HarDBlock_this:[%d]' % (i_blk, ch_out_HarDBlock_this))
                # idx_blk: [0], ch_out_HarDBlock_this: [48]
                # idx_blk: [1], ch_out_HarDBlock_this: [78]
                # idx_blk: [2], ch_out_HarDBlock_this: [160]
                # idx_blk: [3], ch_out_HarDBlock_this: [214]
                # idx_blk: [4], ch_out_HarDBlock_this: [286]
        #end



        # completed to set
        #       <<for FC-HarDNet>>
        #         self.base[]
        #         self.shortcut_layers[]: having module index (in ModuleList) for shortcut (= here, skip_connection)
        #         skip_connection_channel_counts[]: having output ch of each HDB block (not transition-down layer)
        #       <<for RPNet-backbone>>
        #         self.set_idx_module_HarDBlock[]
        # ---------------------------------------------------------------------------------------------------------
        # note that
        #       skip_connection_channel_counts[] is for creating modules in the below (inside __init__()).
        #       self.shortcut_layers[] is used in forward().
        #---------------------------------------------------------------------------------------------------------
        # created HarDBlock & ConvL having
        #       idx_blk:[0], HarDBlock<ch_in: [48], ch_out: [48]>, ConvL<ch_in: [48], ch_out: [64]>
        #       idx_blk:[1], HarDBlock<ch_in: [64], ch_out: [78]>, ConvL<ch_in: [78], ch_out: [96]>
        #       idx_blk:[2], HarDBlock<ch_in: [96], ch_out: [160]>, ConvL<ch_in: [160], ch_out: [160]>
        #       idx_blk:[3], HarDBlock<ch_in: [160], ch_out: [214]>, ConvL<ch_in: [214], ch_out: [224]>
        #       idx_blk:[4], HarDBlock<ch_in: [224], ch_out: [286]>, ConvL<ch_in: [286], ch_out: [320]>
        #---------------------------------------------------------------------------------------------------------
        # To see the whole structure,
        #       please see FC_hardnet at /home/yu1/Desktop/avin_seg/seg_hardnet/
        #---------------------------------------------------------------------------------------------------------
        # message (2020/5/25)
        #   please, check shortcut
        #---------------------------------------------------------------------------------------------------------


        ###================================================================================================
        ### up-network
        ###================================================================================================
        cur_channels_count  = ch            # cur_channels_count: 320
        prev_block_channels = ch            # prev_block_channels: 320
        n_blocks            = blks-1        # n_blocks: 4, blks: 5
        self.n_blocks       = n_blocks

        ###
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up    = nn.ModuleList([])

        ###
        for i in range(n_blocks-1,-1,-1):
            #print('-'*50)
            #print('i:[%d]' % i)

            ###------------------------------------------------------------------------------
            ### append transition-up in self.transUpBlocks[]
            ###------------------------------------------------------------------------------
            self.transUpBlocks.append( TransitionUp(prev_block_channels, prev_block_channels) )         # APPEND
            #print('  prev_block_channels:[%d]' % prev_block_channels)


            ###------------------------------------------------------------------------------
            ### append conv1x1 in self.conv1x1_up[]
            ###------------------------------------------------------------------------------
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append( ConvLayer(cur_channels_count, cur_channels_count//2, kernel=1) )    # APPEND
            #print('  cur_channels_count:[%d], skip_connection_channel_counts:[%d]' % (cur_channels_count, skip_connection_channel_counts[i]))
            #print('  cur_channels_count//2:[%d]' % (cur_channels_count//2))


            ###------------------------------------------------------------------------------
            ### append HarDBlock in self.denseBlocksUp[]
            ###------------------------------------------------------------------------------
            cur_channels_count = cur_channels_count//2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            self.denseBlocksUp.append(blk)                                      # APPEND


            ###------------------------------------------------------------------------------
            ### shift
            ###------------------------------------------------------------------------------
            prev_block_channels = blk.get_out_ch()
            #print('  [in] cur_channels_count: [%d]' % cur_channels_count)
            #print('  [out] blk.get_out_ch():[%d]' % prev_block_channels)

            cur_channels_count = prev_block_channels
        #end


        ###================================================================================================
        ### final conv (for FC-HarDNet)
        ###================================================================================================
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)


        ###================================================================================================
        ### relu for outcome of final conv (for rpnet)
        ###================================================================================================
        self.relu_on_finalConv = nn.ReLU(inplace=True)


        ###================================================================================================
        ### rpnet_decoder_cnvs (for rpnet)
        ###================================================================================================
        #ch_in_rpnet_decoder_cnvs = ch_output_fc_hardnet_a + ch_output_fc_hardnet_b
        #self.rpnet_decoder_cnvs = ConvLayer(in_channels=ch_in_rpnet_decoder_cnvs, out_channels=256, kernel=3)


        ###================================================================================================
        ### rpnet_decoder_hmap_center (for rpnet)
        ###================================================================================================
        ch_in_rpnet_decoder_centerline = ch_output_fc_hardnet_a + ch_output_fc_hardnet_b

        self.rpnet_decoder_centerline = nn.Conv2d(in_channels=ch_in_rpnet_decoder_centerline,
                                                  out_channels=1, kernel_size=1, stride=1,
                                                  padding=0, bias=True)

        if self.n_classes == 4:
            self.rpnet_decoder_AFM           = MyDecoder(in_channels=ch_in_rpnet_decoder_centerline,out_channels=1)


        if n_channels_reg == 3:
            self.rpnet_decoder_leftright  = nn.Conv2d(in_channels=ch_in_rpnet_decoder_centerline,
                                                      out_channels=2, kernel_size=1, stride=1,
                                                      padding=0, bias=True)


        #------------------------------------------------------------------------------
        # Through the above, the followings are set:
        #   self.base
        #   self.transUpBlocks
        #   self.denseBlocksUp
        #   self.conv1x1_up
        #   self.finalConv
        #
        #   self.rpnet_decoder_hmap_center
        #------------------------------------------------------------------------------

        #a = 1
    #end

    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    ############################################################################################################
    ### hardnet::forward()
    ############################################################################################################
    def forward(self, x):

        ###================================================================================================
        ### init
        ###================================================================================================

        ###
        skip_connections = []
        size_in = x.size()


        ###================================================================================================
        ### down-network
        ###================================================================================================
        for idx_module in range(len(self.base)):
            ###------------------------------------------------------------------------------
            ###
            ###------------------------------------------------------------------------------
            x = self.base[idx_module](x)


            ###------------------------------------------------------------------------------
            ### (For FC-HarDNet)
            ###------------------------------------------------------------------------------
            if idx_module in self.shortcut_layers:
                skip_connections.append(x)
            #end
        #end


        ###================================================================================================
        ### up-network (for FC-HarDNet)
        ###================================================================================================
        out_seg = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out_seg = self.transUpBlocks[i](out_seg, skip, True)
            out_seg = self.conv1x1_up[i](out_seg)
            out_seg = self.denseBlocksUp[i](out_seg)
        #end


        ###================================================================================================
        ### insert semantic segmentation output (HDB-3U outcome) to backbone_rpnet
        ###================================================================================================
        backbone_rpnet = out_seg


        ###================================================================================================
        ### [FC-HarDNet] final conv
        ###================================================================================================
        out_seg = self.finalConv(out_seg)


        ###================================================================================================
        ### [rpnet] insert semantic segmentation output (final conv outcome) to backbone_rpnet
        ###================================================================================================
        out_seg_after_relu = self.relu_on_finalConv(out_seg)
        backbone_rpnet = torch.cat([backbone_rpnet, out_seg_after_relu], 1)



        out_centerline = self.rpnet_decoder_centerline(backbone_rpnet)
        ###================================================================================================
        ### [rpnet] rpnet_decoder_attraction_field_map
        ###================================================================================================
        if self.n_classes == 4:
            out_AFM        = self.rpnet_decoder_AFM(backbone_rpnet)
            out_AFM_final  = F.interpolate(out_AFM, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)


        if self.n_channels_reg == 3:
            out_leftright  = self.rpnet_decoder_leftright(backbone_rpnet)


        ###================================================================================================
        ### [FC-HarDNet] interpolate for final result
        ###================================================================================================
        out_seg_final = F.interpolate(out_seg, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)



        ###================================================================================================
        ### [rpnet] interpolate for final result
        ###================================================================================================
        out_centerline_final = F.interpolate(out_centerline, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)


        ###================================================================================================
        ### [rpnet] interpolate for final result attraction field map
        ###================================================================================================
        if self.n_channels_reg == 3:
            out_leftright_final = F.interpolate(out_leftright, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)


        if self.n_channels_reg == 3:
            return out_seg_final, out_centerline_final, out_leftright_final
        elif self.n_channels_reg == 1:
            if self.n_classes == 4:
                return out_seg_final, out_centerline_final, out_AFM_final
            else:
                return out_seg_final, out_centerline_final
    #end




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
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
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


class DinkNet34_less_pool(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet34_more_dilate, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.dblock = Dblock_more_dilate(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DinkNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DinkNet101(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet101, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


#####################################################################################################################
# ERF-Net
#####################################################################################################################
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
        self.pool.padding = (s1,s2)

        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

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

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
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
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes, num_channels):
        super().__init__()
        self.n_channels_reg = num_channels

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.finalconvSeg       = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

        self.finalconvCent = nn.Conv2d(16+num_classes, 1, 1, stride=1, padding=0, bias=True)

        if num_classes == 4:
            self.finalconvAFM =  MyDecoder(in_channels=16+num_classes, out_channels=1)

        if num_channels == 3:
            self.finalconvLR = nn.Conv2d(16+num_classes, 2, 1, stride=1, padding=0, bias=True)


    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)


        backbone   = output
        outseg     = self.finalconvSeg(backbone)
        outsegrelu = F.relu(outseg)
        backbone   = F.interpolate(backbone, size=(540, 960), mode="bilinear", align_corners=True)

        if outsegrelu.shape[2] != backbone.shape[2] or outsegrelu.shape[3] != backbone.shape[3]:
            outsegrelu = F.interpolate(outsegrelu, size=(540, 960), mode="bilinear", align_corners=True)
            outseg     = F.interpolate(outseg, size=(540, 960), mode="bilinear", align_corners=True)

        backbone_erfnet = torch.cat([backbone, outsegrelu], 1)

        outcent_final = self.finalconvCent(backbone_erfnet)

        out_AFM = self.finalconvAFM(backbone_erfnet)
        out_AFM_final = F.interpolate(out_AFM, size=(540, 960), mode="bilinear", align_corners=True)

        outseg_final = outseg

        if self.n_channels_reg == 1:
            return outseg_final, outcent_final, out_AFM_final
        elif self.n_channels_reg == 3:
            outLR_final = self.finalconvLR(backbone_erfnet)
            return outseg_final, outcent_final, outLR_final


# ERFNet
class ERFNet(nn.Module):
    def __init__(self, n_classes=19, n_channels_reg=3):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(num_classes=n_classes, num_channels=n_channels_reg)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)  # predict=False by default
            return self.decoder.forward(output)



##########################################################################
# BiSeNet V2
##########################################################################
class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
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
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
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
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
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
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
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
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        # self.up1 = nn.Upsample(scale_factor=4)
        # self.up2 = nn.Upsample(scale_factor=4)
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        # right1 = self.up1(right1)

        size_in = left1.size()
        right1 =  F.relu(F.interpolate(right1, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True))

        # left = left1 * torch.sigmoid(right1)
        left = left1 * right1
        right = left2 * F.relu(right2)
        # right = self.up2(right)

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
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=1, mode='bilinear', align_corners=False)
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
            nn.Upsample(scale_factor=1, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv_out(x)
        return feat


class Bisenet_v2(nn.Module):
    def __init__(self, n_classes=19, n_channels_reg=3):
        super().__init__()
        self.n_channels_reg = n_channels_reg
        self.n_classes_seg  = n_classes

        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()
        self.head = SegmentHead(128, 1024, n_classes, up_factor=8)
        # self.head = SegmentationHead(128, n_classes)
        self.aux2   = SegmentHead(16, 128, n_classes, up_factor=4)
        self.aux3   = SegmentHead(32, 128, n_classes, up_factor=8)
        self.aux4   = SegmentHead(64, 128, n_classes, up_factor=16)
        self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32)

        self.finalconvCent = nn.Conv2d(128+n_classes, 1, 1, stride=1, padding=0, bias=True)

        if n_channels_reg == 3:
            self.finalconvLR = nn.Conv2d(128+n_classes, 2, 1, stride=1, padding=0, bias=True)

        if n_classes == 4:
            self.finalconvAFM =  MyDecoder(in_channels=128+n_classes, out_channels=1)

    def forward(self,x):

        size_in = x.size()
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)

        outseg    = self.head(feat_head)

        logits_aux2 = self.aux2(feat2)
        logits_aux3 = self.aux3(feat3)
        logits_aux4 = self.aux4(feat4)
        logits_aux5_4 = self.aux5_4(feat5_4)

        outseg_relu = F.relu(outseg)
        backbone = torch.cat([feat_head, outseg_relu], 1)

        out_centerline = self.finalconvCent(backbone)

        if self.n_channels_reg == 3:
            out_leftright  = self.finalconvLR(backbone)
            out_leftright_final = F.interpolate(out_leftright, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)

        if self.n_classes_seg == 4:
            out_AFM = self.finalconvAFM(backbone)
            out_AFM_final = F.interpolate(out_AFM, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)

        outseg_final = F.interpolate(outseg, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)

        out_aux1 = F.interpolate(logits_aux2, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)
        out_aux2 = F.interpolate(logits_aux3, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)
        out_aux3 = F.interpolate(logits_aux4, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)
        out_aux4 = F.interpolate(logits_aux5_4, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)

        out_centerline_final = F.interpolate(out_centerline, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)
        
        if self.n_channels_reg == 3:
            return outseg_final, out_centerline_final, out_leftright_final 
        elif self.n_channels_reg == 1:
            if self.n_classes_seg == 4:
                return outseg_final, out_centerline_final, out_AFM_final
            else:
                return outseg_final, out_centerline_final



    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # self.load_pretrain()


    def load_pretrain(self):
        state = modelzoo.load_url(backbone_url)
        for name, child in self.named_children():
            if name in state.keys():
                child.load_state_dict(state[name], strict=True)

##########################################################################
# Segformer
##########################################################################

import torch
from einops import rearrange
from torch import nn
from torchvision.ops import StochasticDepth
from typing import List
from typing import Iterable
import torch.nn.functional as F



class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

class OverlapPatchMerging(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False
            ),
            LayerNorm2d(out_channels)
        )


class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio
            ),
            LayerNorm2d(channels),
        )
        self.att = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)
        # attention needs tensor of shape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")
        out = self.att(x, reduced_x, reduced_x)[0]
        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out

class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # dense layer
            nn.Conv2d(channels, channels, kernel_size=1),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
            ),
            nn.GELU(),
            # dense layer
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )

class ResidualAdd(nn.Module):
    """Just an util layer"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x

class SegFormerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = .0
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                )
            ),
        )

class SegFormerEncoderStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: List[int],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, overlap_size,
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)


def chunks(data: Iterable, sizes: List[int]):
    """
    Given an iterable, returns slices using sizes as indices
    """
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk


class SegFormerEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            widths: List[int],
            depths: List[int],
            all_num_heads: List[int],
            patch_sizes: List[int],
            overlap_sizes: List[int],
            reduction_ratios: List[int],
            mlp_expansions: List[int],
            drop_prob: float = .0
    ):
        super().__init__()
        # create drop paths probabilities (one for each stage's block)
        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                [in_channels, *widths],
                widths,
                patch_sizes,
                overlap_sizes,
                chunks(drop_probs, sizes=depths),
                depths,
                reduction_ratios,
                all_num_heads,
                mlp_expansions
            )
            ]
        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class SegFormerDecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )


class SegFormerDecoder(nn.Module):
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )

    def forward(self, features):
        new_features = []
        for feature, stage in zip(features, self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features


class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(), # why relu? Who knows
            nn.BatchNorm2d(channels) # why batchnorm and not layer norm? Idk
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x        = torch.cat(features, dim=1)
        x_fuse   = self.fuse(x)
        out_seg  = self.predict(x_fuse)
        return x_fuse,out_seg

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

class SegFormer(nn.Module):
    def __init__(self,n_classes = 19, n_channels_reg = 3):
        in_channels=3
        widths=[64, 128, 256, 512]
        depths=[3, 4, 6, 3]
        all_num_heads=[1, 2, 4, 8]
        patch_sizes=[7, 3, 3, 3]
        overlap_sizes=[4, 2, 2, 2]
        reduction_ratios=[8, 4, 2, 1]
        mlp_expansions=[4, 4, 4, 4]
        decoder_channels=256
        scale_factors=[8, 4, 2, 1]
        drop_prob = 0.0

        self.num_classes = n_classes
        self.n_channels_reg = n_channels_reg

        super().__init__()
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = SegFormerDecoder(decoder_channels, widths[::-1], scale_factors)
        self.head = SegFormerSegmentationHead(
            decoder_channels, n_classes, num_features=len(widths)
        )
        self.relu_on_finalLayer = nn.ReLU(inplace=True)
        self.decoder_centerline = nn.Conv2d(in_channels=decoder_channels+n_classes,
                                                  out_channels=1, kernel_size=1, stride=1,
                                                  padding=0, bias=True)
        
        if n_channels_reg == 3:
            self.finalconvLR = nn.Conv2d(decoder_channels+n_classes, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)
        
        if n_classes == 4:
            self.rpnet_decoder_AFM = MyDecoder(in_channels=decoder_channels+n_classes, out_channels=1)
        

    def forward(self, x):
        size_in = x.size()

        features           = self.encoder(x)
        features           = self.decoder(features[::-1])
        backbone, out_seg  = self.head(features)

        out_seg_after_relu = self.relu_on_finalLayer(out_seg)
        backbone_segformer = torch.cat([backbone, out_seg_after_relu], 1)
        centerline         = self.decoder_centerline(backbone_segformer)

        if self.n_channels_reg == 3:
            out_leftright  = self.finalconvLR(backbone)
            out_leftright_final = F.interpolate(out_leftright, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)

        if self.num_classes == 4:
            AFM = self.rpnet_decoder_AFM(backbone_segformer)
            AFM = F.interpolate(AFM, size=(540, 960), mode="bilinear", align_corners=True)
        
        segmentation = F.interpolate(out_seg, size=(540, 960), mode="bilinear", align_corners=True)
        centerline   = F.interpolate(centerline, size=(540, 960), mode="bilinear",align_corners=True)
       
        if self.n_channels_reg == 3:
            return segmentation, centerline, out_leftright_final 
        elif self.n_channels_reg == 1:
            if self.num_classes == 4:
                return segmentation, centerline, AFM
            else:
                return segmentation, centerline

########################################################################################################################
### main()
########################################################################################################################
if __name__ == '__main__':
    ###
    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)
    #end


    ###
    net = TPEnet_a(n_classes=3).cuda()


    ###
    with torch.no_grad():
        out_seg, out_centerline, out_leftright = net(torch.randn(2, 3, 540, 960).cuda())
        #y = net(torch.randn(2, 3, 512, 512).cuda())
        #y = net(torch.randn(2, 3, 1024, 512).cuda())
        print("out_seg.size()")
        print(out_seg.size())
        print("out_centerline.size()")
        print(out_centerline.size())
        print("out_labelmap_leftright.size()")
        print(out_leftright.size())



