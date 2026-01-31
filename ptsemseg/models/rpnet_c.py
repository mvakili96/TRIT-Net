import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, dilation_this = 1):
        super().__init__()
        if dilation_this == 1:
            self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                              stride=stride, padding=kernel//2, bias = False, dilation = dilation_this))
            self.add_module('norm', nn.BatchNorm2d(out_channels))
            self.add_module('relu', nn.ReLU(inplace=True))
        else:
            self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                                padding='same', bias = False, dilation = dilation_this))
            self.add_module('norm', nn.BatchNorm2d(out_channels))
            self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []

        out_channels = growth_rate

        link = []

        for i in range(10):
            dv = 2 ** i

            if layer % dv == 0:
                k = layer - dv
                link.append(k)

                if i > 0:
                    out_channels *= grmul

        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0

        for i in link:
            ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch

        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, have_dilation = False):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0       

        for i in range(n_layers):
            outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out

            if have_dilation:
                layers_.append( ConvLayer(inch, outch, dilation_this=1) )
            else:
                layers_.append(ConvLayer(inch, outch))


            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch

        self.layers = nn.ModuleList(layers_)

    def forward(self, x):

        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []

            for i in link:
                tin.append(layers_[i])

            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
 
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)

        out_ = []

        for i in range(t):
            if (i == 0 and self.keepBase) or \
               (i == t-1) or (i%2 == 1):
                out_.append(layers_[i])

        out = torch.cat(out_, 1)
        return out


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, x, skip, concat=True):
        out = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode="bilinear",
                align_corners=True)

        if concat:                            
            out = torch.cat([out, skip], 1)
          
        return out


class MyUpSampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, h_new, w_new):
        out = F.interpolate(
            x,
            size=(h_new, w_new),
            mode="bilinear",
            align_corners=True)

        return out


class MyDecoder(nn.Sequential):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=16,
                                          kernel_size=(1, 1), stride=1, padding=0, bias=True))
        self.add_module('norm', nn.BatchNorm2d(16))
        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('conv_b', nn.Conv2d(in_channels=16, out_channels=48,
                                            kernel_size=(3, 3), stride=1, padding=0, bias=True))
        self.add_module('norm_b', nn.BatchNorm2d(48))
        self.add_module('relu_b', nn.ReLU(inplace=True))

        self.add_module('conv_c', nn.Conv2d(in_channels=48, out_channels=out_channels,
                                            kernel_size=(1, 1), stride=1, padding=0, bias=True))

    def forward(self, x):
        return super().forward(x)


class rpnet_c(nn.Module):
    def __init__(self, n_classes_seg = 19):
        super(rpnet_c, self).__init__()

        self.n_classes_seg  = n_classes_seg

        first_ch = [16, 24, 32, 48]
        ch_list  = [64, 96, 160, 224, 320]
        grmul    = 1.7
        gr       = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]
        blks = len(n_layers)           

        ch_output_fc_hardnet_a = 48        
        ch_output_fc_hardnet_b = self.n_classes_seg         

        self.rpnet_decoder_hmap_center = None

        self.base = nn.ModuleList([])

        self.base.append( ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2, dilation_this=1) )  
        self.base.append( ConvLayer(first_ch[0], first_ch[1],  kernel=3, dilation_this=1) )                          
        self.base.append( ConvLayer(first_ch[1], first_ch[2],  kernel=3, stride=2, dilation_this=1) )                
        self.base.append( ConvLayer(first_ch[2], first_ch[3],  kernel=3, dilation_this=1) )                          

        self.shortcut_layers           = []     
        self.set_idx_module_HarDBlock  = []     
        skip_connection_channel_counts = []

        ch = first_ch[3]                        

        for i_blk in range(blks):

            if i_blk == blks-1:
                blk = HarDBlock(ch, gr[i_blk], grmul, n_layers[i_blk], have_dilation=True)
            else:
                blk = HarDBlock(ch, gr[i_blk], grmul, n_layers[i_blk])


            ch = blk.get_out_ch()
            ch_out_HarDBlock_this = blk.get_out_ch()

            skip_connection_channel_counts.append(ch)

            self.base.append( blk )                                              

            self.set_idx_module_HarDBlock.append( len(self.base)-1 )              

            if i_blk < (blks-1):
                self.shortcut_layers.append( len(self.base)-1 )

            self.base.append( ConvLayer(ch, ch_list[i_blk], kernel=1) )           
            ch = ch_list[i_blk]

            if i_blk < blks-1:
                self.base.append( nn.AvgPool2d(kernel_size=2, stride=2) )       

        ###================================================================================================
        ### up-network
        ###================================================================================================
        cur_channels_count  = ch            
        prev_block_channels = ch           
        n_blocks            = blks-1        
        self.n_blocks       = n_blocks

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up    = nn.ModuleList([])

        for i in range(n_blocks-1,-1,-1):

            self.transUpBlocks.append( TransitionUp(prev_block_channels, prev_block_channels) )        

            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append( ConvLayer(cur_channels_count, cur_channels_count//2, kernel=1) )   

            cur_channels_count = cur_channels_count//2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            self.denseBlocksUp.append(blk)                                     

            prev_block_channels = blk.get_out_ch()

            cur_channels_count = prev_block_channels

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=self.n_classes_seg, kernel_size=1, stride=1,
                                   padding=0, bias=True)

        self.relu_on_finalConv = nn.ReLU(inplace=True)

        ch_in_rpnet_decoder_centerline = ch_output_fc_hardnet_a + ch_output_fc_hardnet_b

        self.rpnet_decoder_centerline = nn.Conv2d(in_channels=ch_in_rpnet_decoder_centerline,
                                                  out_channels=1, kernel_size=1, stride=1,
                                                  padding=0, bias=True)
        if self.n_classes_seg == 4:
            self.rpnet_decoder_AFM = MyDecoder(in_channels=ch_in_rpnet_decoder_centerline, out_channels=1)

    def forward(self, x):
        skip_connections = []
        size_in = x.size()

        ###================================================================================================
        ### down-network
        ###================================================================================================
        for idx_module in range(len(self.base)):
            x = self.base[idx_module](x)

            if idx_module in self.shortcut_layers:
                skip_connections.append(x)

        ###================================================================================================
        ### up-network 
        ###================================================================================================
        out_seg = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out_seg = self.transUpBlocks[i](out_seg, skip, True)
            out_seg = self.conv1x1_up[i](out_seg)
            out_seg = self.denseBlocksUp[i](out_seg)

        backbone_rpnet = out_seg

        out_seg = self.finalConv(out_seg)
        out_seg_after_relu = self.relu_on_finalConv(out_seg)
        backbone_rpnet = torch.cat([backbone_rpnet, out_seg_after_relu], 1)

        out_centerline = self.rpnet_decoder_centerline(backbone_rpnet)

        if self.n_classes_seg == 4:
            out_AFM = self.rpnet_decoder_AFM(backbone_rpnet)
            out_AFM_final = F.interpolate(out_AFM, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)

        out_seg_final = F.interpolate(out_seg, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)
        out_centerline_final = F.interpolate(out_centerline, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)

        if self.n_classes_seg == 4:
            return out_seg_final, out_centerline_final, out_AFM_final

        elif self.n_classes_seg == 3:
            return out_seg_final, out_centerline_final



