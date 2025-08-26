import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from core.models.modules.decoders.segformer_head import SegFormerHead
from core.models.utils.modules import init_weight

class ASPP_Classifier_V2(nn.Module):
    def __init__(self, in_channels, dilation_series, padding_series, num_classes):
        super(ASPP_Classifier_V2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(
                    in_channels,
                    num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                )
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x, size=None):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out

class SegFormerHead_Classifier(nn.Module):
    def __init__(self,align_corners,channels,num_classes,in_channels):
        super(SegFormerHead_Classifier, self).__init__()

        self.align_corners = align_corners
        # BN_op = getattr(nn, decoder.settings.norm_layer)
        channels = channels
        self.decoder = SegFormerHead(in_channels=in_channels)
        self.classifier = nn.Conv2d(channels, num_classes, 1, 1)
        init_weight(self.classifier)

    def forward(self, x, size=None):
        #size = (x.shape[2], x.shape[3])  x为最初始的图片形状
        output = self.decoder(x)
        out = {}
        out['embeddings'] = output
        output = self.classifier(output)
        out['pre_logits'] = output
        if size is not None:
            out['logits'] = F.interpolate(output, size=size, mode='bilinear', align_corners=self.align_corners)
            return out['logits']
        else:
            return output
