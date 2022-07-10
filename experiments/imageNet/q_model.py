from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy

from opt_quant.schemes.act import pact
from opt_quant.utils import QConv2d, QLinear, bn



__all__ = [
    "ResNet",
    "resnet18",
    "resnet50",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1,
            w_qmodule=nn.Identity(), act_qmodule=nn.Identity(), err_qmodule=nn.Identity()) -> QConv2d:
    """3x3 convolution with padding"""
    return QConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
            w_qmodule=deepcopy(w_qmodule),
            act_qmodule=deepcopy(act_qmodule),
            err_qmodule=deepcopy(err_qmodule),
        )



def conv1x1(in_planes: int, out_planes: int, stride: int = 1,
            w_qmodule=nn.Identity(), act_qmodule=nn.Identity(), err_qmodule=nn.Identity()) -> QConv2d:
    """1x1 convolution"""
    return QConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                w_qmodule=deepcopy(w_qmodule),
                act_qmodule=deepcopy(act_qmodule),
                err_qmodule=deepcopy(err_qmodule))


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        w_qmodule=nn.Identity(),
        act_qmodule=nn.Identity(),
        err_qmodule=nn.Identity(),
        bn_w_qmodule=nn.Identity(),
        bn_act_qmodule=nn.Identity(),
        bn_err_qmodule=nn.Identity(),
        **kwargs
    ) -> None:
        super().__init__()

        bn_type = kwargs['bn']

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,
                             w_qmodule=w_qmodule, act_qmodule=act_qmodule, err_qmodule=err_qmodule)
        self.bn1 = bn(bn_type, size=planes,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule),
                      bn_err_qmodule=deepcopy(bn_err_qmodule))

        if isinstance(act_qmodule, pact):
            self.relu = nn.Identity()
        else:
            self.relu = nn.ReLU()

        self.conv2 = conv3x3(planes, planes, w_qmodule=w_qmodule, act_qmodule=act_qmodule, err_qmodule=err_qmodule)
        self.bn2 = bn(bn_type, size=planes,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule),
                      bn_err_qmodule=deepcopy(bn_err_qmodule))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out +  identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        w_qmodule=nn.Identity(),
        act_qmodule=nn.Identity(),
        err_qmodule=nn.Identity(),
        bn_w_qmodule=nn.Identity(),
        bn_act_qmodule=nn.Identity(),
        bn_err_qmodule=nn.Identity(),
        **kwargs
    ) -> None:
        super().__init__()
        bn_type = kwargs['bn']
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, w_qmodule=w_qmodule, act_qmodule=act_qmodule, err_qmodule=err_qmodule)
        # self.bn1 = nn.BatchNorm2d(width)
        self.bn1 = bn(bn_type, size=width,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule),
                      bn_err_qmodule=deepcopy(bn_err_qmodule))
        self.conv2 = conv3x3(width, width, stride, groups, dilation,
                             w_qmodule=w_qmodule, act_qmodule=act_qmodule, err_qmodule=err_qmodule)
        # self.bn2 = nn.BatchNorm2d(width)
        self.bn2 = bn(bn_type, size=width,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule),
                      bn_err_qmodule=deepcopy(bn_err_qmodule))
        self.conv3 = conv1x1(width, planes * self.expansion,
                             w_qmodule=w_qmodule, act_qmodule=act_qmodule, err_qmodule=err_qmodule)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3 = bn(bn_type, size=planes * self.expansion,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule),
                      bn_err_qmodule=deepcopy(bn_err_qmodule))

        if isinstance(act_qmodule, pact):
            self.relu = nn.Identity()
        else:
            self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        first_w_qmodule=nn.Identity(), first_act_qmodule=nn.Identity(), first_err_qmodule=nn.Identity(),
        w_qmodule=nn.Identity(), act_qmodule=nn.Identity(), err_qmodule=nn.Identity(),
        bn_w_qmodule=nn.Identity(), bn_act_qmodule=nn.Identity(), bn_err_qmodule=nn.Identity(),
        **kwargs
        ) -> None:
        super(ResNet, self).__init__()

        self.bn_type = kwargs['bn']

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element tuple, got %s" % replace_stride_with_dilation
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = QConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,
                             w_qmodule=deepcopy(first_w_qmodule),
                             act_qmodule=deepcopy(first_act_qmodule),
                             err_qmodule=deepcopy(first_err_qmodule))
        self.bn1 = bn(self.bn_type, size=self.inplanes,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule),
                      bn_err_qmodule=deepcopy(bn_err_qmodule))

        if isinstance(act_qmodule, pact):
            self.relu = nn.Identity()
        else:
            self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       w_qmodule=deepcopy(w_qmodule),
                                       act_qmodule=deepcopy(act_qmodule),
                                       err_qmodule=deepcopy(err_qmodule),
                                       bn_w_qmodule=bn_w_qmodule,
                                       bn_act_qmodule=bn_act_qmodule,
                                       bn_err_qmodule=bn_err_qmodule,
                                       **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       w_qmodule=deepcopy(w_qmodule),
                                       act_qmodule=deepcopy(act_qmodule),
                                       err_qmodule=deepcopy(err_qmodule),
                                       bn_w_qmodule=bn_w_qmodule,
                                       bn_act_qmodule=bn_act_qmodule,
                                       bn_err_qmodule=bn_err_qmodule,
                                       **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       w_qmodule=deepcopy(w_qmodule),
                                       act_qmodule=deepcopy(act_qmodule),
                                       err_qmodule=deepcopy(err_qmodule),
                                       bn_w_qmodule=bn_w_qmodule,
                                       bn_act_qmodule=bn_act_qmodule,
                                       bn_err_qmodule=bn_err_qmodule,
                                       **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       w_qmodule=deepcopy(w_qmodule),
                                       act_qmodule=deepcopy(act_qmodule),
                                       err_qmodule=deepcopy(err_qmodule),
                                       bn_w_qmodule=bn_w_qmodule,
                                       bn_act_qmodule=bn_act_qmodule,
                                       bn_err_qmodule=bn_err_qmodule,
                                       **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, QConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        **kwargs
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride,
                        w_qmodule=kwargs['w_qmodule'],
                        act_qmodule=kwargs['act_qmodule'],
                        err_qmodule=kwargs['err_qmodule']),
                bn(self.bn_type, size=planes * block.expansion,
                      bn_w_qmodule=deepcopy(kwargs['bn_w_qmodule']),
                      bn_act_qmodule=deepcopy(kwargs['bn_act_qmodule']),
                      bn_err_qmodule=deepcopy(kwargs['bn_err_qmodule'])),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                **kwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    first_w_qmodule=nn.Identity(), first_act_qmodule=nn.Identity(), first_err_qmodule=nn.Identity(),
    w_qmodule=nn.Identity(), act_qmodule=nn.Identity(), err_qmodule=nn.Identity(),
    bn_w_qmodule=nn.Identity(), bn_act_qmodule=nn.Identity(), bn_err_qmodule=nn.Identity(),
    **kwargs: Any) -> ResNet:
    model = ResNet(block, layers,
                   first_w_qmodule=first_w_qmodule, first_act_qmodule=first_act_qmodule, first_err_qmodule=first_err_qmodule,
                   w_qmodule=w_qmodule, act_qmodule=act_qmodule, err_qmodule=err_qmodule,
                   bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
                   **kwargs)
    return model
#
# import torchvision
# _COMMON_META = {
#     "min_size": (1, 1),
#     "categories": [i for i in range(1000)],
# }
#
#
def resnet18(*, weights = None, progress: bool = True,
             first_w_qmodule=nn.Identity(), first_act_qmodule=nn.Identity(), first_err_qmodule=nn.Identity(),
             w_qmodule=nn.Identity(), act_qmodule=nn.Identity(), err_qmodule=nn.Identity(),
             bn_w_qmodule=nn.Identity(), bn_act_qmodule=nn.Identity(), bn_err_qmodule=nn.Identity(),
             **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    return _resnet(BasicBlock, [2, 2, 2, 2],
                   first_w_qmodule=first_w_qmodule, first_act_qmodule=first_act_qmodule, first_err_qmodule=first_err_qmodule,
                   w_qmodule=w_qmodule, act_qmodule=act_qmodule, err_qmodule=err_qmodule,
                   bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
                   **kwargs)


def resnet50(*, weights = None, progress: bool = True,
             first_w_qmodule=nn.Identity(), first_act_qmodule=nn.Identity(), first_err_qmodule=nn.Identity(),
             w_qmodule=nn.Identity(), act_qmodule=nn.Identity(), err_qmodule=nn.Identity(),
             bn_w_qmodule=nn.Identity(), bn_act_qmodule=nn.Identity(), bn_err_qmodule=nn.Identity(),
             **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """

    return _resnet(Bottleneck, [3, 4, 6, 3],
                   first_w_qmodule=first_w_qmodule, first_act_qmodule=first_act_qmodule, first_err_qmodule=first_err_qmodule,
                   w_qmodule=w_qmodule, act_qmodule=act_qmodule, err_qmodule=err_qmodule,
                   bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
                   **kwargs)

if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    from torchsummary import summary

    RN50 = resnet50(bn='BN')


    summary(RN50, (256, 3, 224, 224))