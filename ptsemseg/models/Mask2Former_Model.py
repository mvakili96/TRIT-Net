import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

# --------------------- Core Components ---------------------
class SwinBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.window_size = window_size

    def forward(self, x):
        b, c, h, w = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h))
        h_pad, w_pad = h + pad_h, w + pad_w
        h_windows, w_windows = h_pad // self.window_size, w_pad // self.window_size
        x = rearrange(x, 'b c (h1 h2) (w1 w2) -> (h1 w1) (b h2 w2) c',
                     h1=h_windows, w1=w_windows, h2=self.window_size, w2=self.window_size)
        x = self.attn(x, x, x)[0]
        x = self.norm(x)
        x = rearrange(x, '(h1 w1) (b h2 w2) c -> b c (h1 h2) (w1 w2)',
                     h1=h_windows, w1=w_windows, h2=self.window_size, w2=self.window_size, b=b)
        return x[:, :, :h, :w]

class SwinTransformer(nn.Module):
    def __init__(self, embed_dim, depths, num_heads, window_size, drop_path_rate, out_indices):
        super().__init__()
        self.window_size = window_size
        self.strides = [4, 2, 2, 2]
        self.out_indices = out_indices
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=3 if i == 0 else embed_dim * (2 ** (i-1)),
                    out_channels=embed_dim * (2 ** i),
                    kernel_size=3,
                    stride=self.strides[i],
                    padding=1
                ),
                nn.GroupNorm(16, embed_dim * (2 ** i)),
                *[SwinBlock(
                    embed_dim=embed_dim * (2 ** i),
                    num_heads=num_heads[i],
                    window_size=self._adjusted_window_size(i)
                ) for _ in range(depths[i])]
            ) for i in range(4)
        ])

    def _adjusted_window_size(self, stage_idx):
        return 9 if stage_idx == 1 else self.window_size

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                features.append(x)
        return features

# --------------------- Pixel Decoder ---------------------
class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(self,
                 in_channels,        # list of C_i per stage
                 conv_dim=256,
                 num_heads=4,
                 num_encoder_layers=4,
                 dim_feedforward=512,
                 dropout=0.1):
        super().__init__()
        # 1×1 projections to unify channels
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, conv_dim, 1),
                nn.GroupNorm(32, conv_dim),
                nn.ReLU(inplace=True)
            ) for ch in in_channels
        ])
        # 3×3 smoothing convs after fusion
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, 3, padding=1),
                nn.GroupNorm(32, conv_dim),
                nn.ReLU(inplace=True)
            ) for _ in in_channels
        ])
        # Deformable attention across scales
        self.multi_scale_attn = MSDeformAttn(
            embed_dim=conv_dim,
            num_heads=num_heads,
            num_levels=len(in_channels),
            num_points=4
        )
        # Encoder on coarsest tokens only
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, features: list[Tensor]):
        # 1. Project feature maps
        proj_feats = [p(f) for p, f in zip(self.proj, features)]
        # 2. FPN‐style top‐down fusion
        fused = []
        x = proj_feats[-1]
        fused.append(self.smooth[-1](x))
        for i in range(len(proj_feats)-2, -1, -1):
            x = F.interpolate(x, size=proj_feats[i].shape[-2:], mode='nearest')
            x = x + proj_feats[i]
            fused.insert(0, self.smooth[i](x))
        # 3. Flatten each fused map for deformable attention
        bs = fused[0].shape[0]
        tokens = [f.view(bs, f.shape[1], -1).permute(2,0,1) for f in fused]
        # 4. Deformable attention => (sum_hw, B, C)
        context = self.multi_scale_attn(tokens, spatial_shapes=None, level_start_index=None)
        # 5. Extract coarse-level tokens (last level)
        hw = [f.shape[-2]*f.shape[-1] for f in fused]
        start = sum(hw[:-1]); end = sum(hw)
        coarse = context[start:end]  # (Hc*Wc, B, C)
        # 6. Encoder on coarse tokens
        enc = self.encoder(coarse)   # (Hc*Wc, B, C)
        # 7. Map back to coarse feature map
        h_c, w_c = fused[-1].shape[-2:]
        mask_coarse = enc.permute(1,2,0).view(bs, -1, h_c, w_c)
        # 8. Upsample mask features to highest resolution
        h0, w0 = fused[0].shape[-2:]

        # mask_feats = F.interpolate(mask_coarse, size=(h0,w0), mode='bilinear', align_corners=False)
        return mask_coarse, fused

# Placeholder: real MSDeformAttn should be imported from Detectron2
class MSDeformAttn(nn.Module):
    def __init__(self, embed_dim, num_heads, num_levels, num_points):
        super().__init__()
    def forward(self, inputs, spatial_shapes=None, level_start_index=None):
        return torch.cat(inputs, dim=0)

# --------------------- Transformer Decoder ---------------------
class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(self, in_channels, num_queries, num_heads, dropout,
                 dim_feedforward, decoder_layers, pre_norm):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, in_channels)
        layer = nn.TransformerDecoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.output_proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, multi_scale_feats: list[Tensor], mask_feats: Tensor):

        bs = mask_feats.shape[0]
        # flatten multi-scale features into memory
        mem = [f.flatten(2).permute(2,0,1) for f in multi_scale_feats]
        memory = torch.cat(mem, dim=0)  # (sum_hw, B, C)
        # prepare query embeddings as tgt
        tgt = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # (Q, B, C)
        # transformer decoding (cross-attn + self-attn + FFN)
        hs = self.decoder(tgt, memory)
        # reshape queries to spatial map of highest-res
        h0, w0 = mask_feats.shape[-2:]
        out = hs.permute(1,2,0).contiguous().view(bs, -1, h0, w0)
        return self.output_proj(out)


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





# --------------------- Full Model ---------------------
class Mask2FormerMultiMap(nn.Module):
    def __init__(self, n_classes_seg=19, n_channels_reg=3):
        super().__init__()
        self.backbone = SwinTransformer(
            embed_dim=96,
            depths=[2,2,6,2],
            num_heads=[3,6,12,24],
            window_size=15,
            drop_path_rate=0.2,
            out_indices=[0,1,2,3]
        )
        self.pixel_decoder = MSDeformAttnPixelDecoder(
            in_channels=[96,192,384,768],
            conv_dim=256,
            num_heads=4,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer_decoder = MultiScaleMaskedTransformerDecoder(
            in_channels=256,
            num_queries=510,
            num_heads=4,
            dropout=0.1,
            dim_feedforward=1024,
            decoder_layers=3,
            pre_norm=True
        )
        self.shared_upsample = nn.Sequential(
            nn.Conv2d(256,128,3,padding=1),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(128,64,3,padding=1),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(64,32,3,padding=1),
            nn.Upsample(size=(544,960),mode='bilinear',align_corners=False)
        )

            
        blk1 = HarDBlock(256, 24, 1.7, 8)
        cur_channels_count1 = blk1.get_out_ch()
        blk2 = HarDBlock(cur_channels_count1, 18, 1.7, 8)
        cur_channels_count2 = blk2.get_out_ch()
        blk3 = HarDBlock(cur_channels_count2, 16, 1.7, 4)
        cur_channels_count3 = blk3.get_out_ch()

        self.my_upsample = nn.Sequential(
            blk1,
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            blk2,
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            blk3,
            nn.Upsample(size=(544,960),mode='bilinear',align_corners=False)
        )

        self.seg_head = nn.Conv2d(2*cur_channels_count3, n_classes_seg, 1)
        self.relu = nn.ReLU(inplace=True)
        self.centerline_head = nn.Conv2d(2*cur_channels_count3+n_classes_seg, 1, 1)
        self.side_head = nn.Conv2d(2*cur_channels_count3+n_classes_seg, 2 if n_channels_reg==3 else 1, 1)

    def forward(self, x: Tensor):
        feats       = self.backbone(x)
        mask_feats, multi_feats = self.pixel_decoder(feats)
        dec_feats   = self.transformer_decoder(multi_feats, mask_feats)


        # Upsample both pixel and transformer streams
        mask_up = self.my_upsample(multi_feats[0])
        dec_up  = self.my_upsample(dec_feats)

        # Fuse for final mask prediction
        fused_mask = torch.cat([mask_up, dec_up], dim=1)    # (B, 64, H, W)
        seg        = self.seg_head(fused_mask)  # (B, n_classes, H, W)
        seg_after_relu = self.relu(seg)

        # Other heads unchanged
        fused = torch.cat([fused_mask, seg_after_relu], dim=1)
        center = self.centerline_head(fused)
        side   = self.side_head(fused)
        return seg, center, side
