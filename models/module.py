#from attr import s
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("..")
from utils import local_pcd
from models.sync_batchnorm import SynchronizedBatchNorm2d
def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)



class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]
    
    with torch.no_grad():
        # print("src_proj",src_proj)
        # print("ref_proj",ref_proj)
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        # print("rot_xyz",rot_xyz.shape)
        # print("depth_values",depth_values.shape)
        # print("numdepth",num_depth)
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,-1)  # [B, 3, Ndepth, H*W]
        #rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]

        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xyz[:,2:3,:,:][proj_xyz[:, 2:3, :, :] == 0] += 0.0001 # WHY BUG
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    
    warped_src_fea = warped_src_fea.type(torch.float32)
    return warped_src_fea

class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x
class fSEModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(fSEModule, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        reduction = 16
        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features, low_features):
        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        b, c, _, _ = features.size()
        y = self.avg_pool(features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        y = self.sigmoid(y)
        features = features * y.expand_as(features)

        return self.relu(self.conv_se(features))
def upsample(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")
class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet"):
        super(FeatureNet, self).__init__()
        assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        # self.conv0 = nn.Sequential(
        #     Conv2d(3, base_channels, 3, 1, padding=1),
        #     Conv2d(base_channels, base_channels, 3, 1, padding=1),
        # )

        # self.conv1 = nn.Sequential(
        #     Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
        #     Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        #     Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        # )

        # self.conv2 = nn.Sequential(
        #     Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
        #     Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        #     Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        #  )
        
        self.conv0 = nn.Sequential(
            Bneck(3, 3, 12, base_channels, nn.ReLU(True), SE_Module(8), s=1),
            Bneck(3,base_channels, 24,base_channels,nn.ReLU(True), SE_Module(8), s=1),
        )

        self.conv1 = nn.Sequential(
            Bneck(5,base_channels,    24,base_channels * 2, nn.ReLU(True), SE_Module(16),s=2),
            Bneck(3,base_channels * 2,48, base_channels * 2, nn.ReLU(True), SE_Module(16), s=1),
            Bneck(3,base_channels * 2,48, base_channels * 2, nn.ReLU(True), SE_Module(16), s=1),
        )
        
        self.conv2 = nn.Sequential(
            Bneck(5,base_channels * 2, 48, base_channels * 4, nn.ReLU(True), SE_Module(32), s=2),
            Bneck(3,base_channels * 4, 96,base_channels * 4, nn.ReLU(True), SE_Module(32), s=1),
            Bneck(3,base_channels * 4, 96,base_channels * 4, nn.ReLU(True), SE_Module(32), s=1),
         )
        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]
        #final_chs = base_channels * 4
        #self.sa=SpatialAttention(32,32,droprate=0.15)
        self.CAG = CAG(in_channels=32)
        #self.fSE=fSEModule()
        if self.arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out_channels.append(2 * base_channels)
        elif self.arch_mode == "fpn":
            final_chs = base_channels * 4
            if num_stage == 3:
                print("num_stage",num_stage)
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                print("num_stage",num_stage)
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels)

    def forward(self, x):
        #print("x",x.shape)
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        # print("cov0",conv0.shape)
        # print("cov1",conv1.shape)
        # print("cov2",conv2.shape)
        intra_feat = conv2
        #print("intra",intra_feat.shape)
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
      
        if self.arch_mode == "unet":
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out
        
        elif self.arch_mode == "fpn":
            #print("self.num_stage",self.num_stage)
            if self.num_stage == 3:
    
                # intra_feat1 = F.interpolate(intra_feat, scale_factor=2, mode="nearest") 
                # intra_feat=(intra_feat1+self.inner1(conv1))/2
                # CA = self.CAG(intra_feat)
                # out=intra_feat*CA
                # out = self.out2(out)
                # outputs["stage2"] = out
                # intra_feat= F.interpolate(intra_feat, scale_factor=2, mode="nearest") 
                # intra_feat=(intra_feat+self.inner2(conv0))/2
                # CA = self.CAG(intra_feat)
                # out= F.adaptive_max_pool2d(intra_feat, output_size=x.shape[-2:])
                # out=out*CA
                # out = self.out3(out)
                # outputs["stage3"] = out
               
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out
                # a=F.interpolate(intra_feat, scale_factor=2, mode="nearest")
                # b=self.inner1(conv1)
                # print("a",a.shape)
                # print("b",b.shape)
                
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
              
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
               
                # intra_feat1 = F.interpolate(intra_feat, scale_factor=2, mode="nearest") 
                # intra_feat=(intra_feat1+self.inner1(conv1))/2
               
                # CA = self.CAG(intra_feat)
               
                # out=intra_feat*CA
                # out = self.out2(out)
    
                # outputs["stage2"] = out
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out


        return outputs
class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)
        self.spatial= SpatialAttention(kernel_size=3)
    def forward(self, x):
        #print(x.shape)
        conv0 = self.conv0(x)
        #print(conv0.shape)
        conv0=self.spatial(conv0)
       # print(conv0.shape)
        conv2 = self.conv2(self.conv1(conv0))
        #print(conv2.shape)
        conv2=self.spatial(conv2)
        conv4 = self.conv4(self.conv3(conv2))
        #print(conv4.shape)
        conv4=self.spatial(conv4)
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        #print(x.shape)
        return x

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth

def LLSR_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

    return total_loss, depth_loss


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1,
                                                                                               1) * new_interval.unsqueeze(1))

    return depth_range_samples


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=192.0, min_depth=0.0):
    #shape: (B, H, W)
    #cur_depth: (B, H, W) or (B, D)
    #return depth_range_samples: (B, D, H, W)
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )

        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)

    else:

        depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)

    return depth_range_samples


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x+3, inplace=True) / 6
        return out
class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out
class SE_Module(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SE_Module, self).__init__()
 
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True),
 
            hsigmoid(),
        )
 
    def forward(self, x):
        #print("x",x.shape)
        return x * self.se(x)
class Bneck(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, s):
        super(Bneck, self).__init__()
 
        self.stride = s
        self.se = semodule
 
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=s,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
 
        self.shortcut = nn.Sequential()
        if s == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )
 
    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # print("out.shape",out.shape)
        if self.se != None:
            out = self.se(out)
        out += self.shortcut(x) if self.stride == 1 else out
        return out
def calDepthHypo(ref_depths,ref_intrinsics,src_intrinsics,ref_extrinsics,src_extrinsics,depth_min,depth_max):
    ## Calculate depth hypothesis maps for refine steps

    nhypothesis_init = 48
    d = 4
    pixel_interval = 1

    nBatch = ref_depths.shape[0]
    height = ref_depths.shape[1]
    width = ref_depths.shape[2]

    with torch.no_grad():

        ref_depths = ref_depths
        
       
        src_intrinsics = [s[:,:3,:3]for s in src_intrinsics]
        ref_intrinsics = ref_intrinsics[:,:3,:3] 
        # ref_extrinsics = ref_extrinsics[:,:3,:4]
        src_extrinsics =  [s[:,:3,:4]for s in src_extrinsics]
        # print(" src_intrinsics", src_intrinsics)
        # print(" src_extrinsics",src_extrinsics)
       #src_extrinsics=src_extrinsics.double()
        ref_intrinsics = ref_intrinsics.double()
        ref_extrinsics = ref_extrinsics.double()
        # print(" ref_intrinsics", ref_intrinsics)
        # print(" ref_extrinsics",ref_extrinsics)
        interval_maps = []
        depth_range_samples = ref_depths.unsqueeze(1).repeat(1,d*2,1,1)
        for batch in range(nBatch):
            xx, yy = torch.meshgrid([torch.arange(0,width).cuda(),torch.arange(0,height).cuda()])

            xxx = xx.reshape([-1]).double()
            yyy = yy.reshape([-1]).double()

            X = torch.stack([xxx, yyy, torch.ones_like(xxx)],dim=0)

            D1 = torch.transpose(ref_depths[batch,:,:],0,1).reshape([-1]) # Transpose before reshape to produce identical results to numpy and matlab version.
            D2 = D1+1

            X1 = X*D1
            X2 = X*D2
            ray1 = torch.matmul(torch.inverse(ref_intrinsics[batch]),X1)
            ray2 = torch.matmul(torch.inverse(ref_intrinsics[batch]),X2)

            X1 = torch.cat([ray1, torch.ones_like(xxx).unsqueeze(0).double()],dim=0)
            X1 = torch.matmul(torch.inverse(ref_extrinsics[batch]),X1)
            X2 = torch.cat([ray2, torch.ones_like(xxx).unsqueeze(0).double()],dim=0)
            X2 = torch.matmul(torch.inverse(ref_extrinsics[batch]),X2)

            X1 = torch.matmul(src_extrinsics[batch][0].double(), X1)
            X2 = torch.matmul(src_extrinsics[batch][0].double(), X2)

            X1 = X1[:3]
            X1 = torch.matmul(src_intrinsics[batch][0].double(),X1)
            X1_d = X1[2].clone()
            X1 /= X1_d

            X2 = X2[:3]
            X2 = torch.matmul(src_intrinsics[batch][0].double(),X2)
            X2_d = X2[2].clone()
            X2 /= X2_d

            k = (X2[1]-X1[1])/(X2[0]-X1[0])
            b = X1[1]-k*X1[0]

            theta = torch.atan(k)
            X3 = X1+torch.stack([torch.cos(theta)*pixel_interval,torch.sin(theta)*pixel_interval,torch.zeros_like(X1[2,:])],dim=0)

            A = torch.matmul(ref_intrinsics[batch],ref_extrinsics[batch][:3,:3])
            tmp = torch.matmul(src_intrinsics[batch][0],src_extrinsics[batch][0,:3,:3])
            A = torch.matmul(A,torch.inverse(tmp.double()))

            tmp1 = X1_d*torch.matmul(A,X1)
            tmp2 = torch.matmul(A,X3)

            M1 = torch.cat([X.t().unsqueeze(2),tmp2.t().unsqueeze(2)],axis=2)[:,1:,:]
            M2 = tmp1.t()[:,1:]
            ans = torch.matmul(torch.inverse(M1),M2.unsqueeze(2))
            delta_d = ans[:,0,0]

            interval_maps = torch.abs(delta_d).mean().repeat(ref_depths.shape[2],ref_depths.shape[1]).t()

            for depth_level in range(-d,d):
                depth_range_samples[batch,depth_level+d,:,:] += depth_level*interval_maps
    return depth_range_samples.float()
if __name__ == "__main__":
    # some testing code, just IGNORE it
    import sys
    sys.path.append("../")
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    # MVSDataset = find_dataset_def("colmap")
    # dataset = MVSDataset("../data/results/ford/num10_1/", 3, 'test',
    #                      128, interval_scale=1.06, max_h=1250, max_w=1024)

    MVSDataset = find_dataset_def("dtu_yao")
    num_depth = 48
    dataset = MVSDataset("../data/DTU/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, num_depth, interval_scale=1.06 * 192 / num_depth)

    dataloader = DataLoader(dataset, batch_size=1,drop_last=True)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4]  #(B, N, 3, H, W)
    # imgs = item["imgs"][:, :, :, :, :]
    proj_matrices = item["proj_matrices"]   #(B, N, 2, 4, 4) dim=N: N view; dim=2: index 0 for extr, 1 for intric
    proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :]
    # proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :] * 4
    depth_values = item["depth_values"]     #(B, D)

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_proj = proj_matrices[0], proj_matrices[1:][0]  #only vis first view

    src_proj_new = src_proj[:, 0].clone()
    src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
    ref_proj_new = ref_proj[:, 0].clone()
    ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])

    warped_imgs = homo_warping(src_imgs[0], src_proj_new, ref_proj_new, depth_values)

    ref_img_np = ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255
    cv2.imwrite('../tmp/ref.png', ref_img_np)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        img_np = img_np[:, :, ::-1] * 255

        alpha = 0.5
        beta = 1 - alpha
        gamma = 0
        img_add = cv2.addWeighted(ref_img_np, alpha, img_np, beta, gamma)
        cv2.imwrite('../tmp/tmp{}.png'.format(i), np.hstack([ref_img_np, img_np, img_add])) #* ratio + img_np*(1-ratio)]))

class CAG(nn.Module):
    def __init__(self, in_channels):
        super(CAG, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        self.fc1 = nn.Conv2d(in_channels, in_channels, 1)
        self.fc2 = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fc1 = self.relu(self.fc1(self.avgpool(x)))
        fc2 = self.relu(self.fc2(self.maxpool(x)))
        out = fc1 + fc2
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """ 空间注意力机制 将通道维度通过最大池化和平均池化进行压缩，然后合并，再经过卷积和激活函数，结果和输入特征图点乘

        :param kernel_size: 卷积核大小
        """
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print('x shape', x.shape)
        # (2,512,8,8) -> (2,1,8,8)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # (2,512,8,8) -> (2,1,8,8)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # (2,1,8,8) + (2,1,8,8) -> (2,2,8,8)
        cat = torch.cat([avg_out, max_out], dim=1)
        # (2,2,8,8) -> (2,1,8,8)
        out = self.conv1(cat)
        return x * self.sigmoid(out)

class conv_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride, padding,dilation=(1,1),group=1,bn_act=False,bias=False):
        super(conv_block,self).__init__()
        self.conv = nn.Conv3d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,
                              padding=padding,dilation=dilation, groups=group, bias=bias)
        self.bn = SynchronizedBatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)
        self.use_bn_act = bn_act
    def forward(self,x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)
def volumegatelight(in_channels, kernel_size=3, dilation=[1,1], bias=True):
    return nn.Sequential(
        #MSDilateBlock3D(in_channels, kernel_size, dilation, bias),
        conv3d(in_channels, 1, kernel_size=1, stride=1, bias=bias),
        conv3d(1, 1, kernel_size=1, stride=1)
     )
def conv3d(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(0.0,inplace=True)
    )
