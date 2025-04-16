import torch
import torch.nn as nn

# ==== Basic Blocks ====
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s,
                              padding=k // 2 if p is None else p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        hidden_channels = out_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1)
        self.cv2 = Conv(hidden_channels, out_channels, 3)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()
        self.cv1 = Conv(in_channels, out_channels, 1)
        self.m = nn.ModuleList(Bottleneck(out_channels // 2, out_channels // 2) for _ in range(n))
        self.cv2 = Conv(out_channels + out_channels // 2 * n, out_channels, 1)

    def forward(self, x):
        x = self.cv1(x)
        y = list(torch.chunk(x, 2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))

class SCDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = Conv(in_channels, out_channels, 1, 1)
        self.cv2 = Conv(out_channels, out_channels, 3, 2, g=out_channels, act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        self.cv1 = Conv(in_channels, in_channels // 2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(in_channels * 2, out_channels, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.qkv = Conv(in_channels, in_channels * 2, 1)
        self.proj = Conv(in_channels, in_channels, 1)
        self.pe = Conv(in_channels, in_channels, 3, 1, g=in_channels, act=False)

    def forward(self, x):
        return x

class PSA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cv1 = Conv(channels, channels, 1)
        self.attn = Attention(channels)
        self.ffn = nn.Sequential(
            Conv(channels, channels * 2, 1),
            Conv(channels * 2, channels, 1, act=False)
        )
        self.cv2 = Conv(channels, channels, 1)

    def forward(self, x):
        x = self.cv1(x)
        x = self.attn(x)
        x = self.ffn(x)
        return self.cv2(x)

class C2fCIB(C2f):
    pass  # Placeholder, inherits C2f structure

class YOLOv10(nn.Module):
    def __init__(self, variant='n', num_classes=80):
        super().__init__()
        # Configs for each variant
        config = {
            'n': {'depth': 0.33, 'width': 0.25, 'mc': 1024},
            's': {'depth': 0.33, 'width': 0.50, 'mc': 1024},
            'm': {'depth': 0.67, 'width': 0.75, 'mc': 768},
            'b': {'depth': 0.67, 'width': 1.00, 'mc': 512},
            'l': {'depth': 1.00, 'width': 1.00, 'mc': 512},
            'x': {'depth': 1.00, 'width': 1.25, 'mc': 512},
        }[variant]

        d, w, mc = config['depth'], config['width'], config['mc']

        def ch(c): return int(min(c, mc) * w)
        def n_layers(n): return max(round(n * d), 1)

        self.num_classes = num_classes
        self.num_anchors = 3
        self.num_outputs = (num_classes + 5) * self.num_anchors

        # Backbone
        self.layer0 = Conv(3, ch(64), 3, 2)
        self.layer1 = Conv(ch(64), ch(128), 3, 2)
        self.layer2 = C2f(ch(128), ch(128), n_layers(3))
        self.layer3 = Conv(ch(128), ch(256), 3, 2)
        self.layer4 = C2f(ch(256), ch(256), n_layers(6))
        self.layer5 = SCDown(ch(256), ch(512))
        self.layer6 = C2f(ch(512), ch(512), n_layers(6))
        self.layer7 = SCDown(ch(512), ch(1024))
        self.layer8 = C2f(ch(1024), ch(1024), n_layers(3))
        self.layer9 = SPPF(ch(1024), ch(1024))
        self.layer10 = PSA(ch(1024))

        # Neck
        self.up10 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce10 = Conv(ch(1024), ch(512), 1, 1)
        self.c2f11 = C2fCIB(ch(512) + ch(512), ch(512), n_layers(3))

        self.up12 = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce12 = Conv(ch(512), ch(256), 1, 1)
        self.c2f13 = C2fCIB(ch(256) + ch(256), ch(256), n_layers(3))

        self.down14 = SCDown(ch(256), ch(512))
        self.c2f15 = C2fCIB(ch(512) + ch(512), ch(512), n_layers(3))

        self.down16 = SCDown(ch(512), ch(1024))
        self.c2f17 = C2fCIB(ch(1024) + ch(1024), ch(1024), n_layers(3))

        # Detection heads
        self.detect_small = nn.Conv2d(ch(256), self.num_outputs, 1)
        self.detect_medium = nn.Conv2d(ch(512), self.num_outputs, 1)
        self.detect_large = nn.Conv2d(ch(1024), self.num_outputs, 1)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)
        x10 = self.layer10(x9)
        
        u10 = self.reduce10(self.up10(x10))
        cat10 = torch.cat([u10, x6], dim=1)
        x11 = self.c2f11(cat10)

        u11 = self.reduce12(self.up12(x11))
        cat11 = torch.cat([u11, x4], dim=1)
        x12 = self.c2f13(cat11)

        d12 = self.down14(x12)
        cat12 = torch.cat([d12, x11], dim=1)
        x13 = self.c2f15(cat12)

        d13 = self.down16(x13)
        cat13 = torch.cat([d13, x10], dim=1)
        x14 = self.c2f17(cat13)

        out_small = self.detect_small(x12)
        out_medium = self.detect_medium(x13)
        out_large = self.detect_large(x14)

        return out_small, out_medium, out_large

if __name__ == "__main__":
    model = YOLOv10(variant='n', num_classes=5)
    x = torch.randn(1, 3, 640, 640)
    out = model(x)
    for i, o in enumerate(out):
        print(f"Output {i} shape: {o.shape}")
