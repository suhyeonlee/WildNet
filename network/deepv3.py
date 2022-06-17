"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import logging
import torch
from torch import nn
from network import Resnet
from network.mynn import initialize_weights, Norm2d, Upsample

import torchvision.models as models

from network.cel import get_content_extension_loss
import torchvision


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        print("output_stride = ", output_stride)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            Norm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None, cont_proj_head=0, wild_cont_dict_size=0,
                variant='D16', skip='m1', skip_num=48, args=None):
        super(DeepV3Plus, self).__init__()

        self.args = args

        # loss
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean').cuda()

        # create the wild-content dictionary
        self.cont_proj_head = cont_proj_head
        if wild_cont_dict_size > 0:
            if cont_proj_head > 0:
                self.cont_dict = {}
                self.cont_dict['size'] = wild_cont_dict_size
                self.cont_dict['dim'] = self.cont_proj_head

                self.register_buffer("wild_cont_dict", torch.randn(self.cont_dict['dim'], self.cont_dict['size']))
                self.wild_cont_dict = nn.functional.normalize(self.wild_cont_dict, p=2, dim=0) # C X Q
                self.register_buffer("wild_cont_dict_ptr", torch.zeros(1, dtype=torch.long))
                self.cont_dict['wild'] = self.wild_cont_dict.cuda()
                self.cont_dict['wild_ptr'] = self.wild_cont_dict_ptr
            else:
                raise 'dimension of wild-content dictionary is zero'

        # set backbone
        self.variant = variant
        self.trunk = trunk

        channel_1st = 3
        channel_2nd = 64
        channel_3rd = 256
        channel_4th = 512
        prev_final_channel = 1024
        final_channel = 2048

        if trunk == 'resnet-50':
            resnet = Resnet.resnet50(fs_layer=self.args.fs_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D16':
            os = 16
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            raise 'unknown deepv3 variant: {}'.format(self.variant)

        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.dsn = nn.Sequential(
            nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
            Norm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        initialize_weights(self.dsn)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        if self.cont_proj_head > 0:
            self.proj = nn.Sequential(
                nn.Linear(256, 256, bias=True),
                nn.ReLU(inplace=False),
                nn.Linear(256, self.cont_proj_head, bias=True))
            initialize_weights(self.proj)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        if trunk == 'resnet-50':
            in_channel_list = [0, 0, 64, 256, 512, 1024, 2048]
            out_channel_list = [0, 0, 32, 128, 256,  512, 1024]
        else:
            raise ValueError("Not a valid network arch")


    def forward(self, x, gts=None, aux_gts=None, x_w=None, apply_fs=False):

        x_size = x.size()  # 800
        
        # encoder
        x = self.layer0[0](x)
        if self.training & apply_fs:
            with torch.no_grad():
                x_w = self.layer0[0](x_w)
        x = self.layer0[1](x)
        if self.training & apply_fs:
            x_sw = self.layer0[1](x, x_w) # feature stylization
            with torch.no_grad(): 
                x_w = self.layer0[1](x_w)
        x = self.layer0[2](x)
        x = self.layer0[3](x)
        if self.training & apply_fs:
            with torch.no_grad():
                x_w = self.layer0[2](x_w)
                x_w = self.layer0[3](x_w)
            x_sw = self.layer0[2](x_sw)
            x_sw = self.layer0[3](x_sw)
        
        if self.training & apply_fs:
            x_tuple = self.layer1([x, x_w, x_sw])
            low_level = x_tuple[0]
            low_level_w = x_tuple[1]
            low_level_sw = x_tuple[2]
        else:
            x_tuple = self.layer1([x])
            low_level = x_tuple[0]
        
        x_tuple = self.layer2(x_tuple)
        x_tuple = self.layer3(x_tuple)
        aux_out = x_tuple[0]
        if self.training & apply_fs:
            aux_out_w = x_tuple[1]
            aux_out_sw = x_tuple[2]
        x_tuple = self.layer4(x_tuple)
        x = x_tuple[0]
        if self.training & apply_fs:
            x_w = x_tuple[1]
            x_sw = x_tuple[2]

        # decoder
        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)
        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final1(dec0)
        dec2 = self.final2(dec1)
        main_out = Upsample(dec2, x_size[2:])
        
        if self.training:
            # compute original semantic segmentation loss
            loss_orig = self.criterion(main_out, gts)
            aux_out = self.dsn(aux_out)
            if aux_gts.dim() == 1:
                aux_gts = gts
            aux_gts = aux_gts.unsqueeze(1).float()
            aux_gts = nn.functional.interpolate(aux_gts, size=aux_out.shape[2:], mode='nearest')
            aux_gts = aux_gts.squeeze(1).long()
            loss_orig_aux = self.criterion_aux(aux_out, aux_gts)

            return_loss = [loss_orig, loss_orig_aux]

            if apply_fs:
                x_sw = self.aspp(x_sw)
                dec0_up_sw = self.bot_aspp(x_sw)
                dec0_fine_sw = self.bot_fine(low_level_sw)
                dec0_up_sw = Upsample(dec0_up_sw, low_level_sw.size()[2:])
                dec0_sw = [dec0_fine_sw, dec0_up_sw]
                dec0_sw = torch.cat(dec0_sw, 1)
                dec1_sw = self.final1(dec0_sw)
                dec2_sw = self.final2(dec1_sw)
                main_out_sw = Upsample(dec2_sw, x_size[2:])
                
                with torch.no_grad():
                    x_w = self.aspp(x_w)
                    dec0_up_w = self.bot_aspp(x_w)
                    dec0_fine_w = self.bot_fine(low_level_w)
                    dec0_up_w = Upsample(dec0_up_w, low_level_w.size()[2:])
                    dec0_w = [dec0_fine_w, dec0_up_w]
                    dec0_w = torch.cat(dec0_w, 1)
                    dec1_w = self.final1(dec0_w)
                    dec2_w = self.final2(dec1_w)
                    main_out_w = Upsample(dec2_w, x_size[2:])
                
                if self.args.use_cel:
                    # projected features
                    assert (self.cont_proj_head > 0)
                    proj2 = self.proj(dec1.permute(0,2,3,1)).permute(0,3,1,2)
                    proj2_sw = self.proj(dec1_sw.permute(0,2,3,1)).permute(0,3,1,2)
                    with torch.no_grad():
                        proj2_w = self.proj(dec1_w.permute(0,2,3,1)).permute(0,3,1,2)

                    # compute content extension learning loss
                    loss_cel = get_content_extension_loss(proj2, proj2_sw, proj2_w, gts, self.cont_dict)

                    return_loss.append(loss_cel)
                
                if self.args.use_sel:
                    # compute style extension learning loss
                    loss_sel = self.criterion(main_out_sw, gts)
                    aux_out_sw = self.dsn(aux_out_sw)
                    loss_sel_aux = self.criterion_aux(aux_out_sw, aux_gts)
                    return_loss.append(loss_sel)
                    return_loss.append(loss_sel_aux)
                
                if self.args.use_scr:
                    # compute semantic consistency regularization loss
                    loss_scr = torch.clamp((self.criterion_kl(nn.functional.log_softmax(main_out_sw, dim=1), nn.functional.softmax(main_out, dim=1)))/(torch.prod(torch.tensor(main_out.shape[1:]))), min=0)
                    loss_scr_aux = torch.clamp((self.criterion_kl(nn.functional.log_softmax(aux_out_sw, dim=1), nn.functional.softmax(aux_out, dim=1)))/(torch.prod(torch.tensor(aux_out.shape[1:]))), min=0)
                    return_loss.append(loss_scr)
                    return_loss.append(loss_scr_aux)

            return return_loss
        else:
            return main_out


def DeepR50V3PlusD(args, num_classes, criterion, criterion_aux, cont_proj_head, wild_cont_dict_size):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux, cont_proj_head=cont_proj_head, wild_cont_dict_size=wild_cont_dict_size,
                    variant='D16', skip='m1', args=args)
