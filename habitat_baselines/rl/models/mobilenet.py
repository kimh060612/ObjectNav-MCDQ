"""MobileNet3 for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MobileNetV3Seg', 'get_mobilenet_v3_large_seg', 'get_mobilenet_v3_small_seg']

from habitat_baselines.rl.models.baseline import mobilenet_v3_small_1_0, mobilenet_v3_large_1_0


class BaseModel(nn.Module):
    def __init__(self, nclass, aux=False, backbone='mobilenet', pretrained_base=False, **kwargs):
        super(BaseModel, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.backbone = backbone

        if backbone == 'mobilenetv3_small':
            self.pretrained = mobilenet_v3_small_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'mobilenetv3_large':
            self.pretrained = mobilenet_v3_large_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        else:
            raise RuntimeError("Unknown backnone: {}".format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        if self.backbone in ['mobilenetv3_small', 'mobilenetv3_large']:
            x = self.pretrained.conv1(x)
            x = self.pretrained.layer1(x)
            c1 = self.pretrained.layer2(x)
            c2 = self.pretrained.layer3(c1)
            c3 = self.pretrained.layer4(c2)
            c4 = self.pretrained.layer5(c3)
            if self.backbone == 'efficientnet':
                c4 = self.pretrained.layer6(c4)
        elif self.backbone in ['shufflenet', 'shufflenetv2']:
            x = self.pretrained.conv1(x)
            c1 = self.pretrained.maxpool(x)
            c2 = self.pretrained.stage2(c1)
            c3 = self.pretrained.stage3(c2)
            c4 = self.pretrained.stage4(c3)
        else:
            raise ValueError

        return c1, c2, c3, c4

class MobileNetV3Seg(BaseModel):
    def __init__(self, nclass, aux=False, backbone='mobilenetv3_small', pretrained_base=False, **kwargs):
        super(MobileNetV3Seg, self).__init__(nclass, aux, backbone, pretrained_base, **kwargs)
        mode = backbone.split('_')[-1]
        self.head = _Head(nclass, mode, **kwargs)
        if aux:
            inter_channels = 40 if mode == 'large' else 24
            self.auxlayer = nn.Conv2d(inter_channels, nclass, 1)

    def forward(self, x):
        size = x.size()[2:]
        _, c2, _, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c2)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _Head(nn.Module):
    def __init__(self, nclass, mode='small', norm_layer=nn.BatchNorm2d, **kwargs):
        super(_Head, self).__init__()
        in_channels = 960 if mode == 'large' else 576
        self.lr_aspp = _LRASPP(in_channels, norm_layer, **kwargs)
        self.project = nn.Conv2d(128, nclass, 1)

    def forward(self, x):
        x = self.lr_aspp(x)
        return self.project(x)


class _LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, norm_layer, **kwargs):
        super(_LRASPP, self).__init__()
        out_channels = 128
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )
        self.b1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(13, 13), stride=(4, 5)),  # check it
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2  # check it
        return x

def get_mobilenet_v3_large_seg(num_classes, pretrained_path, pretrained_base=True, **kwargs):
    
    model = MobileNetV3Seg(num_classes, backbone='mobilenetv3_large', pretrained_base=pretrained_base, **kwargs)
    model.load_state_dict(torch.load(pretrained_path))
    return model.eval()

def get_mobilenet_v3_small_seg(num_classes, pretrained_path, pretrained_base=True, **kwargs):
    model = MobileNetV3Seg(num_classes, backbone='mobilenetv3_small', pretrained_base=pretrained_base, **kwargs)
    model.load_state_dict(torch.load(pretrained_path))
    return model.eval()
