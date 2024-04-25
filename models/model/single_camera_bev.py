import torch
import torchvision as tv
from torch import nn
import cv2
import numpy as np
import torch.nn.functional as F
import clip
from .bra_legacy import BiLevelRoutingAttention



def naive_init_module(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return mod


class InstanceEmbedding_offset_y_z(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding_offset_y_z, self).__init__()
        self.neck_new = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_offset_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_z = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms_new)
        naive_init_module(self.me_new)
        naive_init_module(self.m_offset_new)
        naive_init_module(self.m_z)
        naive_init_module(self.neck_new)

    def forward(self, x):
        feat = self.neck_new(x)
        return self.ms_new(feat), self.me_new(feat), self.m_offset_new(feat), self.m_z(feat)


class InstanceEmbedding(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding, self).__init__()
        self.neck = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms)
        naive_init_module(self.me)
        naive_init_module(self.neck)

    def forward(self, x):
        feat = self.neck(x)
        return self.ms(feat), self.me(feat)


class LaneHeadResidual_Instance_with_offset_z(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance_with_offset_z, self).__init__()

        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(size=output_size),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 64, 1),
            ),
        )
        self.head = InstanceEmbedding_offset_y_z(64, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up_new)

    def forward(self, bev_x):
        bev_feat = self.bev_up_new(bev_x)
        return self.head(bev_feat)


class LaneHeadResidual_Instance(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance, self).__init__()

        self.bev_up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 60x 24
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(scale_factor=2),  # 120 x 48
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 32, 1),
            ),

            nn.Upsample(size=output_size),  # 300 x 120
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(16, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                )
            ),
        )

        self.head = InstanceEmbedding(32, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up)

    def forward(self, bev_x):
        bev_feat = self.bev_up(bev_x)
        return self.head(bev_feat)


class FCTransform_(nn.Module):
    def __init__(self, image_featmap_size, space_featmap_size, mode = 'simple'):
        super(FCTransform_, self).__init__()
        ic, ih, iw = image_featmap_size
        sc, sh, sw = space_featmap_size
        self.image_featmap_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        self.fc_transform = nn.Sequential(
            nn.Linear(ih * iw, sh * sw),
            nn.ReLU(),
            nn.Linear(sh * sw, sh * sw),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=sc, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(sc),
            nn.ReLU(), )
        self.residual = Residual(
            module=nn.Sequential(
                nn.Conv2d(in_channels=sc, out_channels=sc, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(sc),
            ))

        self.attention = Residual(module=GlobalAttention(dim = sc, AiA = True))
        self.mode = mode

    def forward(self, x):

        x = x.view(list(x.size()[:2]) + [self.image_featmap_size[1] * self.image_featmap_size[2], ])  
        bev_view = self.fc_transform(x)  
        bev_view = bev_view.view(list(bev_view.size()[:2]) + [self.space_featmap_size[1], self.space_featmap_size[2]])
        bev_view = self.conv1(bev_view)
        out = self.residual(bev_view) # out:[8, 256, 25, 5]

        if self.mode == "origin":
            out = out
        elif self.mode == "simple":
            out = self.attention(out)

        return out


class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super(Residual, self).__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class BEV_LaneDet(nn.Module):  # BEV-LaneDet
    def __init__(self, bev_shape, output_2d_shape, train=True):
        super(BEV_LaneDet, self).__init__()

        self.pan = PANet()
        '''
        self.bb_resnet_34 = nn.Sequential(*list(tv.models.resnet34(pretrained=True).children())[:-2]) 
        self.bb_resnet_50 = nn.Sequential(*list(tv.models.resnet50(pretrained=True).children())[:-2],
                                          nn.Conv2d(2048, 512, kernel_size=1),
                                          nn.BatchNorm2d(512),
                                          nn.ReLU(),
                                          )        
        self.bb_efficientb4 = nn.Sequential(*list(tv.models.efficientnet_b4(pretrained=True).children())[:-2],
                                          nn.Conv2d(1792, 512, kernel_size=1),
                                          nn.BatchNorm2d(512),
                                          nn.ReLU(),
                                          )
        '''
        self.down64 = naive_init_module(
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # S64
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(1024)
                ),
                downsample=nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            )
        )

        '''
        self.down128 = naive_init_module(
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),  # S128
                    nn.BatchNorm2d(2048),
                    nn.ReLU(),
                    nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(2048)
                ),
                downsample=nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),
            )
        )
        '''

        '''
        self.prepocess_448 = nn.Sequential(
                    nn.Conv2d(448, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU()
                )

        self.prepocess_512 = nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=1, stride=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU()
                )
        '''

        self.s32transformer = FCTransform_((512, 18, 32), (256, 25, 5))
        self.s64transformer = FCTransform_((1024, 9, 16), (256, 25, 5))
        # self.s128transformer = FCTransform_((2048, 5, 8), (256, 25, 5))
        self.lane_head = LaneHeadResidual_Instance_with_offset_z(bev_shape, input_channel=512)
        self.is_train = train
        if self.is_train:
            self.lane_head_2d = LaneHeadResidual_Instance(output_2d_shape, input_channel=512)
        self.weight = nn.Parameter(torch.Tensor([0]))


    def forward(self, img):
        # img: [8, 3, 576 ,1024]

        # img_s32 = self.bb_efficientb4(img) # img_s32: [8, 512, 18, 32]

        # img_s32 = self.bb_resnet_50(img)

        img_s32 = self.pan(img)
        img_s64 = self.down64(img_s32)
        #img_s128 = self.down128(img_s64)

        bev_32 = self.s32transformer(img_s32)
        bev_64 = self.s64transformer(img_s64)
        # bev_128 = self.s128transformer(img_s128)

        '''
        w1 = torch.exp(self.weight[0]) / torch.sum(torch.exp(self.weight))
        w2 = torch.exp(self.weight[1]) / torch.sum(torch.exp(self.weight))
        w3 = torch.exp(self.weight[2]) / torch.sum(torch.exp(self.weight))
        bev = w1 * bev_128 + w2 * bev_64 + w3 * bev_32
        '''

        bev = torch.cat([bev_64, bev_32], dim=1)
        # weight = self.weight

        if self.is_train:
            return self.lane_head(bev), self.lane_head_2d(img_s32), #weight, gamma_32, gamma_64
        else:
            return self.lane_head(bev)


class GlobalAttention(nn.Module):
    def __init__(self, dim, AiA = True):
        super(GlobalAttention, self).__init__()
        in_dim = dim // 8
        self.query_conv = nn.Conv2d(in_channels=dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.AIA = AiA

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)

        if self.AIA:
            identity = v_aia = energy
            energy = torch.bmm(energy, energy)
            energy = torch.softmax(energy, dim = -1)
            energy = torch.bmm(v_aia, energy.permute(0, 2, 1))
            energy = energy + identity

        attention = torch.softmax(energy, dim = -1)
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x

        return out




class PANet(nn.Module):
    def __init__(self):
        super(PANet, self).__init__()

        backbone = tv.models.resnet34(pretrained=True)

        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        #self.relu1 = nn.ReLU(inplace=True)
        #self.relu2 = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu1 = backbone.relu
        self.maxpool = backbone.maxpool


        self.layer1 = nn.Sequential(*list(backbone.layer1.children()))
        self.layer2 = nn.Sequential(*list(backbone.layer2.children()))
        self.layer3 = nn.Sequential(*list(backbone.layer3.children()))
        self.layer4 = nn.Sequential(*list(backbone.layer4.children()))

        # Lateral connections
        self.lateral_layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.lateral_layer2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.lateral_layer3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.lateral_layer4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Up connections
        self.up_layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.up_layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.up_layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        #self.shortcut = nn.Conv2d(64, 512, kernel_size=1, stride=8)

        # Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        #self.gamma = nn.Parameter(torch.Tensor([0]))

    def forward(self, x):
        # [8,3,576,1024]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x) # [8,64,288,512] 

        layer1 = self.layer1(x)      # [8,64,144,256] 
        #short_cut = self.shortcut(layer1)
        layer2 = self.layer2(layer1) # [8,128,72,128] 
        layer3 = self.layer3(layer2) # [8,256,36,64] 
        layer4 = self.layer4(layer3) # [8,512,18,32] 

        fusion_layer4 = self.lateral_layer4(layer4)                                 # [8,512,18,32] 
        fusion_layer3 = layer3 + self.upsample(self.lateral_layer3(fusion_layer4))  # [8,256,36,64]
        fusion_layer2 = layer2 + self.upsample(self.lateral_layer2(fusion_layer3))  # [8,128,72,128] 
        fusion_layer1 = layer1 + self.upsample(self.lateral_layer1(fusion_layer2))  # [8,64,144,256] 

        pan_layer1 = fusion_layer1                               # [8,64,144,256] 
        pan_layer2 = fusion_layer2 + self.up_layer1(pan_layer1)  # [8,128,72,128] 
        pan_layer3 = fusion_layer3 + self.up_layer2(pan_layer2)  # [8,256,36,64] 
        pan_layer4 = fusion_layer4 + self.up_layer3(pan_layer3)  # [8,512,18,32] 

        pan_layer4 = self.conv2(pan_layer4)
        pan_out = pan_layer4 #+ self.gamma * layer4

        return pan_out



