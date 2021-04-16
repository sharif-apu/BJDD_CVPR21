import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torchsummary import summary

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def swish(x):
    return x * torch.sigmoid(x)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def tensorImgeSplit(targetTensor):
    return targetTensor[:,:1,:,:], targetTensor[:,1:2,:,:], targetTensor[:,2:,:,:]


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class attentionGuidedResBlock(nn.Module):
    def __init__(self, squeezeFilters = 32, expandFilters = 64, dilationRate=1,bn=True, act=False, bias=True):
        super(attentionGuidedResBlock, self).__init__()

        self.bn = bn
        self.act = act
        self.depthAtten = SELayer(squeezeFilters)
        self.exConv = nn.Conv2d(squeezeFilters, expandFilters,  1, padding=0)
        self.exConvBn = nn.BatchNorm2d(expandFilters)
        self.sepConv =  SeparableConv2d(expandFilters, expandFilters,  kernel_size=3,stride=1,dilation=dilationRate,padding=dilationRate)
        self.sepConvBn =  nn.BatchNorm2d(expandFilters)
        self.sqConv = nn.Conv2d(expandFilters, squeezeFilters, 1, padding=0)
        self.sqConvBn = nn.BatchNorm2d(squeezeFilters)
        
    def forward(self, inputTensor):
        xDA = self.depthAtten(inputTensor)
        if self.bn == True:
            xEx = F.relu(self.exConvBn(self.exConv(inputTensor)))
            xSp = F.relu(self.sepConvBn(self.sepConv(xEx)))
            xSq = self.sqConvBn(self.sqConv(xSp))  
        else:
            xEx = F.relu(self.exConv(inputTensor))
            xSp = F.relu(self.sepConv(xEx))
            xSq = self.sqConv(xSp)
        return inputTensor + xSq + xDA


class pixelShuffleUpsampling(nn.Module):
    def __init__(self, inputFilters, scailingFactor=2):
        super(pixelShuffleUpsampling, self).__init__()
    
        self.upSample = nn.Sequential(  nn.Conv2d(inputFilters, inputFilters * (scailingFactor**2), 3, 1, 1),
                                        nn.BatchNorm2d(inputFilters * (scailingFactor**2)),
                                        nn.PixelShuffle(upscale_factor=scailingFactor),
                                        nn.PReLU()
                                    )
    def forward(self, tensor):
        return self.upSample(tensor)






class selfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(selfAttention, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)


#et = attentionGuidedResBlock(dilationRate=4)
#summary(net, input_size = (32, 128, 128))
#print ("reconstruction network")