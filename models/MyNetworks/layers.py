import torch
import torch.nn as nn
import torch.nn.functional as F

class Separabel_conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 kernel_size=(3,3),
                 dilation=(1,1),
                 #padding=(1,1),
                 stride=(1,1),
                 bias=False):
        
        super(Separabel_conv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding= (int((kernel_size[0]-1)/2)*dilation[0],int((kernel_size[1]-1)/2)*dilation[1]),
            dilation= dilation, groups=groups,bias=bias
        )
        self.dw_bn = nn.BatchNorm2d(out_channels)
        self.pointwise_conv2d = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1, padding=0, dilation=1, groups=1, bias=False
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)
    def forward(self, input):

        out = self.depthwise_conv2d(input)
        out = self.dw_bn(out)
        out = self.pointwise_conv2d(out)
        out = self.pw_bn(out)

        return out

# ChannelAttention Mechanism
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        print("------ChannelAttention--------")
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))
        max_out =self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# SpatialAttention Mechanism
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        print("------SpatialAttention--------")
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# GAM_Attention Mechanism
class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

# down_sampling_block
class down_sampling_block(nn.Module):

    def __init__(self, inpc, oupc):
        super(down_sampling_block, self).__init__()
        self.branch_conv = nn.Conv2d(inpc, oupc-inpc, 3, stride=2, padding= 1, bias=False)

        self.branch_mp = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bn = nn.BatchNorm2d(oupc, eps=1e-03)

    def forward(self, x):

        output = torch.cat([self.branch_conv(x), self.branch_mp(x)], 1)

        output = self.bn(output)

        return F.relu(output)

# SE（Squeeze-Excitation）Mechanism
class SE(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        print("------Squeeze-Excitation--------")
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return x*F.sigmoid(out)

# ECA Mechanism
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y) 

        return x * y.expand_as(x)

# bottleneck model
class bt(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 ):
        super(bt, self).__init__()
        self.internal_channels = in_channels // 4
        # compress conv
        self.conv1 = nn.Conv2d(in_channels, self.internal_channels, 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        # a relu
        self.conv2 = nn.Conv2d(self.internal_channels, self.internal_channels, kernel_size,
                               stride, padding=int((kernel_size-1)/2*dilation), dilation=dilation, groups=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(self.internal_channels)
        # a relu
        self.conv4 = nn.Conv2d(self.internal_channels, out_channels, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):

        residual = x
        main = F.relu(self.conv1_bn(self.conv1(x)),inplace=True)
        main = F.relu(self.conv2_bn(self.conv2(main)), inplace=True)
        main = self.conv4_bn(self.conv4(main))
        out = F.relu(torch.add(main, residual), inplace=True)

        return out

# non-bottleneck model
class non_bt(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 kernel_size=3,
                 dilation=1):
        super(non_bt, self).__init__()

        self.conv1 =  nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride, padding=int((kernel_size-1)/2*dilation), dilation=dilation, groups=1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(out_channels)
        # here is relu
        self.conv2 =  nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride, padding=int((kernel_size-1)/2*dilation), dilation=dilation, groups=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        # here is relu
    def forward(self, x):

        x1 = x
        x = F.relu(self.conv1_bn(self.conv1(x)), inplace=True)
        x = self.conv2_bn(self.conv2(x))
        return F.relu(torch.add(x, x1), inplace=True)

# bottleneck model with standard convolution
class BM_SC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 ):
        super(BM_SC, self).__init__()

        #残差分支的卷积
        #3x3
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1,bias=False)
        self.conv4_bn = nn.BatchNorm2d(out_channels)

        self.internal_channels = in_channels // 4
        # compress conv
        self.conv1 = nn.Conv2d(in_channels, self.internal_channels, 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        # a relu
        self.conv2 = nn.Conv2d(self.internal_channels, self.internal_channels, kernel_size,
                               stride, padding=int((kernel_size-1)/2*dilation), dilation=dilation, groups=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(self.internal_channels)
        # a relu
        self.conv3 = nn.Conv2d(self.internal_channels, out_channels, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):

        residual = self.conv4(x)
        residual = self.conv4_bn(residual)

        main = F.relu(self.conv1_bn(self.conv1(x)),inplace=True)
        main = F.relu(self.conv2_bn(self.conv2(main)), inplace=True)
        main = self.conv3_bn(self.conv3(main))
        out = F.relu(torch.add(main, residual), inplace=True)

        return out

# (depthwise-factorized convolution)  DFC
class DFC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 dropout_rate =0.0,
                 ):

        # default decoupled
        super(DFC, self).__init__()

        # Depthwise_conv 3x1 and 1x3 factorized_conv
        self.conv1 = nn.Conv2d(in_channels, in_channels, (kernel_size,1), stride=(stride,1),
                               padding=(int((kernel_size-1)/2*dilation),0), dilation=(dilation,1),
                               groups=in_channels, bias=False)
        self.conv1_bn = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, (1,kernel_size), stride=(1,stride),
                               padding=(0,int((kernel_size-1)/2*dilation)), dilation=(1, dilation),
                               groups=in_channels, bias=False)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        

        self.internal_channels = in_channels // 4                  
        # compress conv
        self.conv3 = nn.Conv2d(in_channels, self.internal_channels, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(self.internal_channels)
        
        # relu

        # Depthwise_conv 3x1 and 1x3 factorized_conv
        self.conv4 = nn.Conv2d(self.internal_channels, self.internal_channels, (kernel_size,1), stride=(stride,1),
                               padding=(int((kernel_size-1)/2*dilation),0), dilation=(dilation,1),
                               groups=self.internal_channels, bias=False)
        self.conv4_bn = nn.BatchNorm2d(self.internal_channels)

        self.conv5 = nn.Conv2d(self.internal_channels, self.internal_channels, (1,kernel_size), stride=(1,stride),
                               padding=(0,int((kernel_size-1)/2*dilation)), dilation=(1, dilation),
                               groups=self.internal_channels, bias=False)
        self.conv5_bn = nn.BatchNorm2d(self.internal_channels)

        self.conv6 = nn.Conv2d(self.internal_channels, out_channels, 1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(out_channels)

        # regularization
        self.dropout = nn.Dropout2d(inplace=True, p=dropout_rate)

        
    def forward(self, input):
        residual = self.conv1(input)
        residual = self.conv1_bn(residual)
        residual = self.conv2(residual)
        residual = self.conv2_bn(residual)

        main = self.conv3(input)
        main = self.conv3_bn(main)
        main = F.relu(main, inplace=True)
        main = self.conv4(main)
        main = self.conv4_bn(main)
        main = self.conv5(main)
        main = self.conv5_bn(main)
        main = self.conv6(main)
        main = self.conv6_bn(main)

        if self.dropout.p != 0:
            residual = self.dropout(residual)

        return F.relu(torch.add(main, residual), inplace=True)

# ( lightweight feature extraction module) LFEM
class LFEM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 dropout_rate =0.0,
                 ):

        # default decoupled
        super(LFEM, self).__init__()

        # Depthwise_conv 3x1 and 1x3 factorized_conv
        self.conv1 = nn.Conv2d(in_channels, in_channels, (kernel_size,1), stride=(stride,1),
                               padding=(int((kernel_size-1)/2*dilation),0), dilation=(dilation,1),
                               groups=in_channels, bias=False)
        self.conv1_bn = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, (1,kernel_size), stride=(1,stride),
                               padding=(0,int((kernel_size-1)/2*dilation)), dilation=(1, dilation),
                               groups=in_channels, bias=False)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        

        self.internal_channels = in_channels // 4                  
        # compress conv
        self.conv3 = nn.Conv2d(in_channels, self.internal_channels, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(self.internal_channels)
        
        # relu

        # Depthwise_conv 3x1 and 1x3 factorized_conv
        self.conv4 = nn.Conv2d(self.internal_channels, self.internal_channels, (kernel_size,1), stride=(stride,1),
                               padding=(int((kernel_size-1)/2*dilation),0), dilation=(dilation,1),
                               groups=self.internal_channels, bias=False)
        self.conv4_bn = nn.BatchNorm2d(self.internal_channels)

        self.conv5 = nn.Conv2d(self.internal_channels, self.internal_channels, (1,kernel_size), stride=(1,stride),
                               padding=(0,int((kernel_size-1)/2*dilation)), dilation=(1, dilation),
                               groups=self.internal_channels, bias=False)

        # -------eca---------
        self.eca = eca_layer(self.internal_channels)
        # -------eca---------

        self.conv5_bn = nn.BatchNorm2d(self.internal_channels)

        self.conv6 = nn.Conv2d(self.internal_channels, out_channels, 1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(out_channels)

        # regularization
        self.dropout = nn.Dropout2d(inplace=True, p=dropout_rate)

        
    def forward(self, input):
        residual = self.conv1(input)
        residual = self.conv1_bn(residual)
        residual = self.conv2(residual)
        residual = self.conv2_bn(residual)

        main = self.conv3(input)
        main = self.conv3_bn(main)
        main = F.relu(main, inplace=True)
        main = self.conv4(main)
        main = self.conv4_bn(main)
        main = self.conv5(main)
     
        main = self.eca(main)

        main = self.conv5_bn(main)
        main = self.conv6(main)
        main = self.conv6_bn(main)

        if self.dropout.p != 0:
            residual = self.dropout(residual)

        return F.relu(torch.add(main, residual), inplace=True)
