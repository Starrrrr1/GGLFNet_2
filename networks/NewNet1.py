import math

import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from timm.models.layers import trunc_normal_


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size=3, padding=1,group=1):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,groups=group),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class AttnConv(nn.Module):
    def __init__(self, in_channels, out_channels,group=None, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if not group:
            group=1
        self.conv1 = Conv(in_channels,mid_channels,kernel_size=3,padding=1)
        self.conv2_1=Conv(mid_channels,out_channels,kernel_size=3,padding=1,group=group)
        self.conv2_2=Conv(mid_channels,out_channels,kernel_size=5,padding=2,group=group)
        self.conv2_3=Conv(mid_channels,out_channels,kernel_size=7,padding=3,group=group)
        self.final_conv=Conv(out_channels,out_channels,kernel_size=3,padding=1)


    def forward(self, x):
        x=self.conv1(x)
        u=x.clone()
        x=self.conv2_1(x)+self.conv2_2(x)+self.conv2_3(x)
        self.final_conv(x)
        x=u+x
        return x

class Down_c(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #OverlapPatchEmbed(in_channels,in_channels),
            AttnConv(in_channels,out_channels)
            #DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Global_attention_Module(nn.Module):
    def __init__(self, dim,L=66836, eps=1e-6,drop_path=0.1, kernel_function=nn.ReLU(),scale=295):
        super().__init__()
        self.pos_emb=nn.Parameter(torch.randn(1, L, dim))
        self.drop_path = DropPath(
                 drop_path) if drop_path > 0. else nn.Identity()
        self.linear_Q = nn.Linear(dim, dim)
        self.linear_K = nn.Linear(dim, dim)
        self.linear_V = nn.Linear(dim, dim)

        self.linear_Q1 = nn.Linear(dim, dim)
        self.linear_K1 = nn.Linear(dim, dim)
        self.scale_c=dim**0.5
        self.scale_L=L**0.5

        self.eps = eps
        self.kernel_fun = kernel_function
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm=nn.LayerNorm(dim)
        self.norm_Q1=nn.LayerNorm(L)
        self.norm_K1=nn.LayerNorm(L)
        #self.act=nn.ReLU()


    def forward(self, x):
        #B,C,H,W=x.shape
        shortcut=x.clone()
        #x=x.permute(0,2,3,1).reshape(B,-1,C).contiguous()
        x=x+self.pos_emb
        Q = self.linear_Q(x)  # blc
        K = self.linear_K(x)  # blc
        V = self.linear_V(x)  # blc
        Q1 = self.linear_Q1(x)
        K1 = self.linear_K1(x)

        Q1 = self.norm_Q1(Q1.transpose(-1, -2)).transpose(-1, -2)
        K1 = self.norm_K1(K1.transpose(-1, -2)).transpose(-1, -2)

        Q = self.kernel_fun(Q)
        K = self.kernel_fun(K)
        Q=Q/self.scale_c
        K = K / self.scale_c

        K = K.transpose(-2, -1)  # bcl
        KV = torch.einsum("bml, blc->bmc", K, V)  # bcc

        Z = 1 / (torch.einsum("blc,bc->bl", Q, K.sum(dim=-1) + self.eps))  # bl

        result = torch.einsum("blc,bcc,bl->blc", Q, KV, Z)  # blc

        Q1 = self.kernel_fun(Q1)
        K1 = self.kernel_fun(K1)
        Q1=Q1/self.scale_L
        K1=K1/self.scale_L



        mid_result = (Q1.transpose(-1, -2) @ K1).softmax(dim=-1)



        result = result @ mid_result

        x = shortcut + self.drop_path(self.gamma * result)
        # x=x.permute(0,2,1).reshape(B,C,H,W).contiguous()
        # x=self.act(self.norm(x))

        return self.norm(x)

# class Global_attention_Module(nn.Module):
#     def __init__(self, dim,L_o=87296,L=300, eps=1e-6,drop_path=0.1, kernel_function=nn.ReLU(),scale=295):
#         super().__init__()
#         self.pool = nn.Linear(L_o,L)
#         self.pos_emb=nn.Parameter(torch.randn(1, L, dim))
#         self.drop_path = DropPath(
#                  drop_path) if drop_path > 0. else nn.Identity()
#         self.linear_Q = nn.Linear(dim, dim)
#         self.linear_K = nn.Linear(dim, dim)
#         self.linear_V = nn.Linear(dim, dim)
#
#         self.linear_Q1 = nn.Linear(dim, dim)
#         self.linear_K1 = nn.Linear(dim, dim)
#         # self.scale_c=dim**0.5
#         # self.scale_L=L**0.5
#
#         self.eps = eps
#         self.kernel_fun = kernel_function
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.norm=nn.LayerNorm(dim)
#         self.norm_Q1=nn.LayerNorm(L)
#         self.norm_K1=nn.LayerNorm(L)
#         self.interpolation=nn.Linear(L,L_o)
#         #self.act=nn.ReLU()
#
#
#     def forward(self, x):
#         #B,C,H,W=x.shape
#         shortcut=x.clone()
#         x=self.pool(x.permute(0,2,1)).permute(0,2,1)
#         #x=x.permute(0,2,3,1).reshape(B,-1,C).contiguous()
#         x=x+self.pos_emb
#         Q = self.linear_Q(x)  # blc
#         K = self.linear_K(x)  # blc
#         V = self.linear_V(x)  # blc
#         Q1 = self.linear_Q1(x)
#         K1 = self.linear_K1(x)
#
#         Q1 = self.norm_Q1(Q1.transpose(-1, -2)).transpose(-1, -2)
#         K1 = self.norm_K1(K1.transpose(-1, -2)).transpose(-1, -2)
#
#         Q = self.kernel_fun(Q)
#         K = self.kernel_fun(K)
#         # Q=Q/self.scale_c
#         # K = K / self.scale_c
#
#         K = K.transpose(-2, -1)  # bcl
#         KV = torch.einsum("bml, blc->bmc", K, V)  # bcc
#
#         Z = 1 / (torch.einsum("blc,bc->bl", Q, K.sum(dim=-1) + self.eps))  # bl
#
#         result = torch.einsum("blc,bcc,bl->blc", Q, KV, Z)  # blc
#
#         Q1 = self.kernel_fun(Q1)
#         K1 = self.kernel_fun(K1)
#         # Q1=Q1/self.scale_L
#         # K1=K1/self.scale_L
#
#
#
#         mid_result = (Q1.transpose(-1, -2) @ K1).softmax(dim=-1)
#
#
#
#         result = result @ mid_result
#         result = self.interpolation(x.permute(0, 2, 1)).permute(0, 2, 1)
#
#
#         x = shortcut + self.drop_path(self.gamma * result)
#         # x=x.permute(0,2,1).reshape(B,C,H,W).contiguous()
#         # x=self.act(self.norm(x))
#
#         return self.norm(x)

# class Global_attention_Module(nn.Module):
#     def __init__(self, dim,L=87296, eps=1e-6,drop_path=0.1, kernel_function=nn.ReLU(),scale=295):


class Global_Branch(nn.Module):
    def __init__(self, dim, block_num=1):
        super().__init__()
        self.blocks=nn.ModuleList()
        for i_block in range(block_num):
            block=Global_attention_Module(dim)
            self.blocks.append(block)
    def forward(self,x):
        shortcut=x.clone()
        for block in self.blocks:
            x=block(x)
        return x

class Global_information(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.conv0=nn.Conv2d(dim,dim,1,1,0)
        self.conv1 = nn.Conv2d(2*dim, dim, 1, 1, 0)
        self.conv2 = nn.Conv2d(4*dim, dim, 1, 1, 0)
        self.conv3 = nn.Conv2d(8*dim, dim, 1, 1, 0)
        self.conv4 = nn.Conv2d(8*dim, dim, 1, 1, 0)
        self.Global_ifo=Global_Branch(dim=dim)

        self.conv0T = nn.Conv2d(dim, dim, 1, 1, 0)
        self.conv1T = nn.Conv2d( dim, 2 *dim, 1, 1, 0)
        self.conv2T = nn.Conv2d( dim, 4 *dim, 1, 1, 0)
        self.conv3T = nn.Conv2d( dim, 8 *dim, 1, 1, 0)
        self.conv4T = nn.Conv2d( dim, 8 *dim, 1, 1, 0)
        self.alpha = nn.Parameter(torch.zeros(5))
        self.norm_relu0=nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.norm_relu1 = nn.Sequential(
            nn.BatchNorm2d(2*dim),
            nn.ReLU(inplace=True)
        )
        self.norm_relu2 = nn.Sequential(
            nn.BatchNorm2d(4 * dim),
            nn.ReLU(inplace=True)
        )
        self.norm_relu3 = nn.Sequential(
            nn.BatchNorm2d(8 * dim),
            nn.ReLU(inplace=True)
        )
        self.norm_relu4 = nn.Sequential(
            nn.BatchNorm2d(8 * dim),
            nn.ReLU(inplace=True)
        )
        self.norm=nn.LayerNorm(dim)
        self.norm1=nn.LayerNorm(dim)
    def forward(self,x):
        B,C,_,_=x[0].shape
        x0,x1,x2,x3,x4=x[0],x[1],x[2],x[3],x[4]
        #print(x0.shape)
        x0 = self.conv0(x0).permute(0,2,3,1).reshape(B,-1,C)
        x1 = self.conv1(x1).permute(0,2,3,1).reshape(B,-1,C)
        x2 = self.conv2(x2).permute(0,2,3,1).reshape(B,-1,C)
        x3 = self.conv3(x3).permute(0,2,3,1).reshape(B,-1,C)
        x4 = self.conv4(x4).permute(0,2,3,1).reshape(B,-1,C)
        t1 = torch.cat([x0, x1, x2, x3, x4], dim=1).contiguous()
        t1=self.norm(t1)
        # #print(t1.shape)
        t1 =t1 + self.Global_ifo(t1)
        t1=self.norm1(t1)
        #print(t1.shape)
        x0 = self.conv0T(t1[:,0:50176,:].permute(0,2,1).reshape(B,C,224,224)).contiguous()
        x1 = self.conv1T(t1[:, 50176:62720, :].permute(0,2,1).reshape(B,C, 112, 112)).contiguous()
        x2 = self.conv2T(t1[:, 62720:65856:, :].permute(0,2,1).reshape(B,C, 56, 56)).contiguous()
        x3 = self.conv3T(t1[:, 65856:66640, :].permute(0,2,1).reshape(B,C, 28, 28)).contiguous()
        x4 = self.conv4T(t1[:, 66640:66836, :].permute(0,2,1).reshape(B,C, 14, 14)).contiguous()

        x[0] = self.norm_relu0(x[0] + self.alpha[0]*x0)
        x[1] = self.norm_relu1(x[1] + self.alpha[1]*x1)
        x[2] = self.norm_relu2(x[2] + self.alpha[2]*x2)
        x[3] = self.norm_relu3(x[3] + self.alpha[3]*x3)
        x[4] = self.norm_relu4(x[4] + self.alpha[4]*x4)

        return x

class GGFuse(nn.Module):
    def __init__(self, in_channels, out_channels,):
        super(GGFuse, self).__init__()
        self.conv_g= nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        #self.act=nn.LeakyReLU(inplace=True)
        self.norm=nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, g, s, d):
        att = torch.sigmoid(self.conv_g(g))

        p_add = self.conv_s(s*att)
        i_add = self.conv_d(d*(1-att))

        return self.norm(self.conv_f(p_add + i_add)+g)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.Fuse=GGFuse(in_channels//3,in_channels//3)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x3=self.Fuse(x3,x2,x1)

        x = torch.cat([x2, x1, x3], dim=1)
        return self.conv(x)

# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()
#
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class NewNet(nn.Module):
    def __init__(self, n_classes, n_channels=3,dim=64, bilinear=True):
        super(NewNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, dim)
        self.down1 = Down_c(dim, dim*2)
        self.down2 = Down_c(dim*2, dim*4)
        self.down3 = Down(dim*4,dim*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(dim * 8, dim*16 // factor)
        self.Global_ifo=Global_information(dim=dim)
        #self.leader=LearderMoudle(dim,dim)
        self.up1 = Up(dim*24, dim*8 // factor, bilinear)
        self.up2 = Up(dim*12, dim*4 // factor, bilinear)
        self.up3 = Up(dim*6, dim*2 // factor, bilinear)
        self.up4 = Up(dim*3, dim, bilinear)
        self.outc = OutConv(dim, n_classes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #print(x3.shape)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_list=[x1,x2,x3,x4,x5]
        x_list=self.Global_ifo(x_list)


        x = self.up1(x5, x4, x_list[3])
        x = self.up2(x, x3, x_list[2])
        x = self.up3(x, x2, x_list[1])
        x = self.up4(x, x1, x_list[0])

        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        logits = self.outc(x)
        return logits