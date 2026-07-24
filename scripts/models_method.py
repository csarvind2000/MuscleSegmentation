"""DixonFuse: physics-informed contrast fusion for thigh-muscle Dixon MRI.

Variants (for ablation):
  wateronly : 1 channel  (Water)                      -> UNet
  dixon4    : 4 channels (W,F,In,Opp)                 -> UNet     [baseline]
  dixon5    : 5 channels (W,F,In,Opp,FF)              -> UNet
  dixonfuse : 5 channels -> ContrastAttention gate -> 1x1 -> UNet [full method]

The encoder is always ImageNet-pretrained ResNet-34 (in_channels=3); a learnable
1x1 stem maps the (gated) input channels to 3. This keeps the backbone identical
across variants so gains are attributable to the fusion, not the backbone.
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ContrastAttention(nn.Module):
    """Squeeze-and-excite gate across input contrast channels (per-image weights)."""
    def __init__(self, in_ch, reduction=1):
        super().__init__()
        hidden = max(2, in_ch // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, in_ch), nn.Sigmoid(),
        )
        self.last_gate = None

    def forward(self, x):
        g = self.fc(x)                    # B x C in [0,1]
        self.last_gate = g.detach()
        return x * g[:, :, None, None]


class DixonFuseUNet(nn.Module):
    def __init__(self, variant, num_classes):
        super().__init__()
        # gateonly: contrast-attention over the 4 Dixon channels, WITHOUT fat-fraction
        # ffsup: fat-fraction is NOT an input; instead an auxiliary head regresses the
        #        per-pixel fat-fraction from the shared decoder, so the physics prior
        #        supervises the representation rather than adding a redundant input channel.
        ch = {"wateronly": 1, "dixon4": 4, "gateonly": 4, "dixon5": 5,
              "dixonfuse": 5, "ffsup": 4}[variant]
        self.variant = variant
        self.in_ch = ch
        self.aux_ff = (variant == "ffsup")           # auxiliary fat-fraction regression
        self.attn = ContrastAttention(ch) if variant in ("dixonfuse", "gateonly") else None
        self.stem = nn.Conv2d(ch, 3, kernel_size=1)
        out_classes = num_classes + (1 if self.aux_ff else 0)   # +1 = FF regression map
        self.unet = smp.Unet("resnet34", encoder_weights="imagenet",
                             in_channels=3, classes=out_classes)

    def select(self, x):
        # x is full 5-channel tensor; pick the channels this variant uses
        if self.in_ch == 1:
            return x[:, 0:1]
        if self.in_ch == 4:
            return x[:, 0:4]
        return x[:, 0:5]

    def forward(self, x):
        x = self.select(x)
        if self.attn is not None:
            x = self.attn(x)
        x = self.stem(x)
        return self.unet(x)    # for ffsup, last channel is the FF regression logit
