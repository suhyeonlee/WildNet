"""
# Code Adapted from:
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#
# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import network.mynn as mynn
from network.adain import AdaptiveInstanceNormalization


__all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class Bottleneck(nn.Module):
    """
    Bottleneck Layer for Resnet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, fs=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = mynn.Norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = mynn.Norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = mynn.Norm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        self.fs = fs
        if self.fs == 1:
            self.instance_norm_layer = AdaptiveInstanceNormalization()
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x_tuple):
        if len(x_tuple) == 1:
            x = x_tuple[0]
        elif len(x_tuple) == 3:
            x = x_tuple[0]
            x_w = x_tuple[1]
            x_sw = x_tuple[2]
        else:
            raise NotImplementedError("%d is not supported length of the tuple"%(len(x_tuple)))

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if len(x_tuple) == 3:
            with torch.no_grad():
                residual_w = x_w

                out_w = self.conv1(x_w)
                out_w = self.bn1(out_w)
                out_w = self.relu(out_w)

                out_w = self.conv2(out_w)
                out_w = self.bn2(out_w)
                out_w = self.relu(out_w)

                out_w = self.conv3(out_w)
                out_w = self.bn3(out_w)

                if self.downsample is not None:
                    residual_w = self.downsample(x_w)

                out_w += residual_w

            residual_sw = x_sw 

            out_sw = self.conv1(x_sw) 
            out_sw = self.bn1(out_sw) 
            out_sw = self.relu(out_sw) 

            out_sw = self.conv2(out_sw) 
            out_sw = self.bn2(out_sw) 
            out_sw = self.relu(out_sw) 

            out_sw = self.conv3(out_sw) 
            out_sw = self.bn3(out_sw) 

            if self.downsample is not None:
                residual_sw = self.downsample(x_sw) 

            out_sw += residual_sw


        if self.fs == 1:
            out = self.instance_norm_layer(out)
            if len(x_tuple) == 3:
                out_sw = self.instance_norm_layer(out_sw, out_w)
                with torch.no_grad():
                    out_w = self.instance_norm_layer(out_w)

        out = self.relu(out)

        if len(x_tuple) == 3:
            with torch.no_grad():
                out_w = self.relu(out_w)
            out_sw = self.relu(out_sw) 
            return [out, out_w, out_sw]
        else:
            return [out]


class ResNet(nn.Module):
    """
    Resnet Global Module for Initialization
    """

    def __init__(self, block, layers, fs_layer=None, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if fs_layer[0] == 1:
            self.bn1 = AdaptiveInstanceNormalization()
            self.relu = nn.ReLU(inplace=False)
        else:
            self.bn1 = mynn.Norm2d(64)
            self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], fs_layer=fs_layer[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, fs_layer=fs_layer[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, fs_layer=fs_layer[3])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, fs_layer=fs_layer[4])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, fs_layer=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                mynn.Norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, fs=0))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                fs=0 if (fs_layer > 0 and index < blocks - 1) else fs_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=True, fs_layer=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if fs_layer is None:
        fs_layer = [0, 0, 0, 0, 0]
    model = ResNet(Bottleneck, [3, 4, 6, 3], fs_layer=fs_layer, **kwargs)
    if pretrained:
        print("########### pretrained ##############")
        mynn.forgiving_state_restore(model, model_zoo.load_url(model_urls['resnet50']))
    return model
