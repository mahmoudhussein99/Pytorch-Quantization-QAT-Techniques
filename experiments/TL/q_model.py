'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_quant.schemes.act import pact

from opt_quant.utils import QConv2d, QLinear, bn



class add_op(nn.Module):
    def __init__(self):
        super(add_op, self).__init__()

    def forward(self, x, y):
        return x+y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,
                 downsample=None,
                 w_qmodule=nn.Identity(),
                 act_qmodule=nn.Identity(),
                 err_qmodule=nn.Identity(),
                 bn_w_qmodule=nn.Identity(), bn_act_qmodule=nn.Identity(), bn_err_qmodule=nn.Identity(),
                 **kwargs
                 ):
        super(BasicBlock, self).__init__()

        bn_type = kwargs['bn']
        shortcut_quant = False
        if 'shortcut_quant' in kwargs.keys() and kwargs['shortcut_quant'] == False:
            shortcut_quant = False

        self.conv1 = QConv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False,
            w_qmodule=deepcopy(w_qmodule),
            act_qmodule=deepcopy(act_qmodule),
            err_qmodule=deepcopy(err_qmodule),
        )
        self.bn1 = bn(bn_type, size=planes,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule), bn_err_qmodule=deepcopy(bn_err_qmodule))


        self.conv2 = QConv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False,
            w_qmodule=deepcopy(w_qmodule),
            act_qmodule=deepcopy(act_qmodule),
            err_qmodule=deepcopy(err_qmodule),
        )
        self.bn2 = bn(bn_type, size=planes,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule),
                      bn_err_qmodule=deepcopy(bn_err_qmodule))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
            if shortcut_quant:
                self.shortcut = nn.Sequential(
                    QConv2d(
                        in_planes, self.expansion * planes,
                        kernel_size=1, stride=stride, bias=False,
                        w_qmodule=deepcopy(w_qmodule),
                        act_qmodule=deepcopy(act_qmodule),
                        err_qmodule=deepcopy(err_qmodule),
                    ),
                    bn(bn_type, size=planes,
                       bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule), bn_err_qmodule=deepcopy(bn_err_qmodule))

                )
        self.add_op = add_op()
        self.downsample=downsample
        if not isinstance(act_qmodule, nn.Identity):
            self.act_fn_1 = nn.Identity()
            self.act_fn_2 = nn.Identity()
        else:
            self.act_fn_1 = nn.ReLU()
            self.act_fn_2 = nn.ReLU()

    def forward(self, x):
        identity=x
        out = self.act_fn_1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity=self.downsample(x)
        out = self.add_op(out, identity)
        out = self.act_fn_2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=101,
                 w_qmodule=nn.Identity(), act_qmodule=nn.Identity(), err_qmodule=nn.Identity(),
                 bn_w_qmodule=nn.Identity(), bn_act_qmodule=nn.Identity(), bn_err_qmodule=nn.Identity(),
                 first_w_qmodule=nn.Identity(), first_act_qmodule=nn.Identity(), first_err_qmodule=nn.Identity(),
                 **kwargs):
        super(ResNet, self).__init__()

        last_layer_quant = False

        bn_type = kwargs['bn']
        self.bnType=bn_type
        if 'last_layer_quant' in kwargs.keys() and kwargs['last_layer_quant']:
            last_layer_quant = True

        self.in_planes = 64

        # Quantize the first layer if user wants!
        self.conv1 = QConv2d(
            3, 64, kernel_size=7,
            stride=2, padding=3, bias=False,
            w_qmodule=deepcopy(first_w_qmodule),
            act_qmodule=deepcopy(first_act_qmodule),
            err_qmodule=deepcopy(first_err_qmodule),
        )

        self.bn1 = bn(bn_type, size=64,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule),
                      bn_err_qmodule=deepcopy(bn_err_qmodule))

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       w_qmodule=deepcopy(w_qmodule),
                                       act_qmodule=deepcopy(act_qmodule),
                                       err_qmodule=deepcopy(err_qmodule),
        bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
                                       **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       w_qmodule=deepcopy(w_qmodule),
                                       act_qmodule=deepcopy(act_qmodule),
                                       err_qmodule=deepcopy(err_qmodule),
        bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
                                       **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       w_qmodule=deepcopy(w_qmodule),
                                       act_qmodule=deepcopy(act_qmodule),
                                       err_qmodule=deepcopy(err_qmodule),
        bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
                                       **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       w_qmodule=deepcopy(w_qmodule),
                                       act_qmodule=deepcopy(act_qmodule),
                                       err_qmodule=deepcopy(err_qmodule),
        bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
                                       **kwargs)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Quantize the last layer if user wants!
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        if last_layer_quant:
            self.linear = QLinear(
                512 * block.expansion, num_classes,
                w_qmodule=deepcopy(w_qmodule),
                act_qmodule=deepcopy(act_qmodule),
                err_qmodule=deepcopy(err_qmodule),
            )

        if not isinstance(act_qmodule, nn.Identity):
            self.act_fn = nn.Identity()
            self.last_act = nn.ReLU()
        else:
            self.act_fn = nn.ReLU()
            self.last_act = nn.Identity()


    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        downsample=None
        #norm_layer=
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                     QConv2d(self.in_planes, planes*block.expansion, kernel_size=1,stride=stride, padding=0, bias=False,
                            w_qmodule=nn.Identity(),act_qmodule=nn.Identity(),err_qmodule=nn.Identity()),
                            bn(self.bnType,planes * block.expansion,bn_w_qmodule=nn.Identity(), 
                                bn_act_qmodule=nn.Identity(),bn_err_qmodule=nn.Identity()),                                                                           )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample,**kwargs) )
        self.in_planes = planes * block.expansion
        for _ in range(1,num_blocks):
            layers.append(block(self.in_planes,planes,1,None,**kwargs))
	  #TODO_M: Edit this for ResNet50	            
        #for stride in strides:
        #    layers.append(block(self.in_planes, planes, stride, **kwargs))
        #    self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act_fn(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = self.last_act(out) #We add this because when we have pact the last ReLu is removed!
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
#M_EDIT
class ResNet_ImageNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000,
                 w_qmodule=nn.Identity(), act_qmodule=nn.Identity(), err_qmodule=nn.Identity(),
                 bn_w_qmodule=nn.Identity(), bn_act_qmodule=nn.Identity(), bn_err_qmodule=nn.Identity(),
                 first_w_qmodule=nn.Identity(), first_act_qmodule=nn.Identity(), first_err_qmodule=nn.Identity(),
                 **kwargs):
        super(ResNet, self).__init__()

        last_layer_quant = False

        bn_type = kwargs['bn']

        if 'last_layer_quant' in kwargs.keys() and kwargs['last_layer_quant']:
            last_layer_quant = True

        self.in_planes = 64

        # Quantize the first layer if user wants!
        self.conv1 = QConv2d(
            3, self.in_planes, kernel_size=7,
            stride=2, padding=3, bias=False,
            w_qmodule=deepcopy(first_w_qmodule),
            act_qmodule=deepcopy(first_act_qmodule),
            err_qmodule=deepcopy(first_err_qmodule),
        )

        self.bn1 = bn(bn_type, size=self.in_planes,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule),
                      bn_err_qmodule=deepcopy(bn_err_qmodule))

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       w_qmodule=deepcopy(w_qmodule),
                                       act_qmodule=deepcopy(act_qmodule),
                                       err_qmodule=deepcopy(err_qmodule),
        bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
                                       **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       w_qmodule=deepcopy(w_qmodule),
                                       act_qmodule=deepcopy(act_qmodule),
                                       err_qmodule=deepcopy(err_qmodule),
        bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
                                       **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       w_qmodule=deepcopy(w_qmodule),
                                       act_qmodule=deepcopy(act_qmodule),
                                       err_qmodule=deepcopy(err_qmodule),
        bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
                                       **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       w_qmodule=deepcopy(w_qmodule),
                                       act_qmodule=deepcopy(act_qmodule),
                                       err_qmodule=deepcopy(err_qmodule),
        bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
                                       **kwargs)
        self.avgpool = nn.AvgPool2d(4)
        # Quantize the last layer if user wants!
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        if last_layer_quant:
            self.fc = QLinear(
                512 * block.expansion, num_classes,
                w_qmodule=deepcopy(w_qmodule),
                act_qmodule=deepcopy(act_qmodule),
                err_qmodule=deepcopy(err_qmodule),
            )

        if not isinstance(act_qmodule, nn.Identity):
            self.act_fn = nn.Identity()
            self.last_act = nn.ReLU()
        else:
            self.act_fn = nn.ReLU()
            self.last_act = nn.Identity()


    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **kwargs))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act_fn(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.last_act(out) #We add this because when we have pact the last ReLu is removed!
        out = self.avgpool(out)
        # out = out.view(out.size(0), -1)
        out= torch.flatten(x,1)
        out = self.fc(out)
        return out


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride,
                 w_qmodule=nn.Identity(),
                 act_qmodule=nn.Identity(),
                 err_qmodule=nn.Identity(),
                 bn_w_qmodule=nn.Identity(), bn_act_qmodule=nn.Identity(), bn_err_qmodule=nn.Identity(),
                 **kwargs
                 ):
        super(Block, self).__init__()
        self.stride = stride

        bn_type = kwargs['bn']
        shortcut_quant = True
        if 'shortcut_quant' in kwargs.keys() and kwargs['shortcut_quant'] == False:
            shortcut_quant = False

        planes = expansion * in_planes
        self.conv1 = QConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False,
                             w_qmodule=deepcopy(w_qmodule),
                             act_qmodule=deepcopy(act_qmodule),
                             err_qmodule=deepcopy(err_qmodule))
        self.bn1 =  bn(bn_type, size=planes,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule), bn_err_qmodule=deepcopy(bn_err_qmodule))
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False,
                               w_qmodule=deepcopy(w_qmodule),
                               act_qmodule=deepcopy(act_qmodule),
                               err_qmodule=deepcopy(err_qmodule)
                               )
        self.bn2 = bn(bn_type, size=planes,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule), bn_err_qmodule=deepcopy(bn_err_qmodule))
        self.conv3 = QConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False,
                               w_qmodule=deepcopy(w_qmodule),
                               act_qmodule=deepcopy(act_qmodule),
                               err_qmodule=deepcopy(err_qmodule)
                               )

        self.bn3 =  bn(bn_type, size=out_planes,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule), bn_err_qmodule=deepcopy(bn_err_qmodule))

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            if shortcut_quant:
                self.shortcut = nn.Sequential(
                    QConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False,
                            w_qmodule=deepcopy(w_qmodule),
                            act_qmodule=deepcopy(act_qmodule),
                            err_qmodule=deepcopy(err_qmodule)
                            ),
                bn(bn_type, size=out_planes,
                       bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule),
                       bn_err_qmodule=deepcopy(bn_err_qmodule)),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

# From https://github.com/kuangliu/pytorch-cifar
class MobileNet_V2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10,
                 w_qmodule=nn.Identity(), act_qmodule=nn.Identity(), err_qmodule=nn.Identity(),
                 bn_w_qmodule=nn.Identity(), bn_act_qmodule=nn.Identity(), bn_err_qmodule=nn.Identity(),
                 first_w_qmodule=nn.Identity(), first_act_qmodule=nn.Identity(), first_err_qmodule=nn.Identity(),
                 **kwargs):
        super(MobileNet_V2, self).__init__()

        last_layer_quant = False

        bn_type = kwargs['bn']


        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = QConv2d(
            3, 32, kernel_size=3, stride=1, padding=1, bias=False,
            w_qmodule=deepcopy(first_w_qmodule),
            act_qmodule=deepcopy(first_act_qmodule),
            err_qmodule=deepcopy(first_err_qmodule),
        )

        self.bn1 = bn(bn_type, size=32,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule),
                      bn_err_qmodule=deepcopy(bn_err_qmodule))

        self.layers = self._make_layers(in_planes=32,
                                        w_qmodule=deepcopy(w_qmodule),
                                        act_qmodule=deepcopy(act_qmodule),
                                        err_qmodule=deepcopy(err_qmodule),
                                        bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule,
                                        bn_err_qmodule=bn_err_qmodule,
                                        **kwargs)

        self.conv2 = QConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False,
                             w_qmodule=deepcopy(w_qmodule),
                             act_qmodule=deepcopy(act_qmodule),
                             err_qmodule=deepcopy(err_qmodule))

        self.bn2 = bn(bn_type, size=1280,
                      bn_w_qmodule=deepcopy(bn_w_qmodule), bn_act_qmodule=deepcopy(bn_act_qmodule),
                      bn_err_qmodule=deepcopy(bn_err_qmodule))

        self.linear = nn.Linear(1280, num_classes)
        if last_layer_quant:
            self.linear = QLinear(1280, num_classes,
                                  w_qmodule=deepcopy(w_qmodule),
                                  act_qmodule=deepcopy(act_qmodule),
                                  err_qmodule=deepcopy(err_qmodule))

    def _make_layers(self, in_planes, **kwargs):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, **kwargs))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def ResNet18(
        num_classes=101,
        w_qmodule=nn.Identity(), act_qmodule=nn.Identity(), err_qmodule=nn.Identity(),
        bn_w_qmodule=nn.Identity(), bn_act_qmodule=nn.Identity(), bn_err_qmodule=nn.Identity(),
        first_w_qmodule=nn.Identity(), first_act_qmodule=nn.Identity(), first_err_qmodule=nn.Identity(),
        **kwargs
):
    return ResNet(
        BasicBlock, [2, 2, 2, 2],num_classes=num_classes,
        w_qmodule=w_qmodule,
        act_qmodule=act_qmodule,
        err_qmodule=err_qmodule,
        bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
        first_w_qmodule=first_w_qmodule, first_act_qmodule=first_act_qmodule, first_err_qmodule=first_err_qmodule,
        **kwargs
    )

def ResNet50(
        num_classes=101,
        w_qmodule=nn.Identity(), act_qmodule=nn.Identity(), err_qmodule=nn.Identity(),
        bn_w_qmodule=nn.Identity(), bn_act_qmodule=nn.Identity(), bn_err_qmodule=nn.Identity(),
        first_w_qmodule=nn.Identity(), first_act_qmodule=nn.Identity(), first_err_qmodule=nn.Identity(),
        **kwargs
):
    return ResNet_ImageNet(
        Block, [3, 4, 6, 3],num_classes=num_classes,
        w_qmodule=w_qmodule,
        act_qmodule=act_qmodule,
        err_qmodule=err_qmodule,
        bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
        first_w_qmodule=first_w_qmodule, first_act_qmodule=first_act_qmodule, first_err_qmodule=first_err_qmodule,
        **kwargs
    )
def MobileNetV2(num_classes=101,w_qmodule=nn.Identity(), act_qmodule=nn.Identity(), err_qmodule=nn.Identity(),
        bn_w_qmodule=nn.Identity(), bn_act_qmodule=nn.Identity(), bn_err_qmodule=nn.Identity(),
        first_w_qmodule=nn.Identity(), first_act_qmodule=nn.Identity(), first_err_qmodule=nn.Identity(),
        **kwargs):
    return MobileNet_V2(num_classes=num_classes,
                        w_qmodule=w_qmodule,
                        act_qmodule=act_qmodule,
                        err_qmodule=err_qmodule,
                        bn_w_qmodule=bn_w_qmodule, bn_act_qmodule=bn_act_qmodule, bn_err_qmodule=bn_err_qmodule,
                        first_w_qmodule=first_w_qmodule, first_act_qmodule=first_act_qmodule,
                        first_err_qmodule=first_err_qmodule,
                        **kwargs
                        )


def test():
    from opt_quant.schemes.weight import sawb
    from opt_quant.schemes.act import lsq_act
    net = ResNet18(bn_quant=True, w_qmodule=sawb(bitwidth=2), bn='BN', act_qmodule=lsq_act(bitwidth=2))
    print(net)
    x = torch.randn(2,3,32,32)
    print(net)
    # torchinfo.summary(net, [2,3,32,32])
    # y = net(x)
    # print(y.size())


if __name__ == '__main__':
    import torchinfo
    test()
