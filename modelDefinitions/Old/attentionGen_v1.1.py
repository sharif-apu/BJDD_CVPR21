import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from basicBlocks import *    

class attentionNet(nn.Module):
    def __init__(self, squeezeFilters = 32, expandFilters = 64, scailingFactor = 2, numAttentionBlock=10):
        super(attentionNet, self).__init__()

        # Input Layer
        self.inputConv = nn.Conv2d(3, squeezeFilters, 3,1,1)
        self.inputConv_bn = nn.BatchNorm2d(squeezeFilters)
        self.featureAttention0 = selfAttention(squeezeFilters, squeezeFilters,3,1,1)
        self.featureAttention0_bn = nn.BatchNorm2d(squeezeFilters)
        
        # Spatial Feature Extraction
        self.globalPooling =  nn.AvgPool2d(2,2) 
        depthAttenBlock = []
        for i in range (numAttentionBlock):
            depthAttenBlock.append(attentionGuidedResBlock(squeezeFilters, expandFilters))
        self.spatialFeatExtBlock = nn.Sequential(*depthAttenBlock)
        self.psUpsampling = pixelShuffleUpsampling(inputFilters=squeezeFilters, scailingFactor=2)
        
        # Full-res Feature Corelation
        self.featureAttention1 = selfAttention(squeezeFilters, squeezeFilters,3,1,1)
        self.featureAttention1_bn = nn.BatchNorm2d(squeezeFilters)
        depthAttenBlock = []
        for i in range (numAttentionBlock//2):
            depthAttenBlock.append(attentionGuidedResBlock(squeezeFilters, expandFilters, bn=False))
        self.fullFeatCorelationBlock = nn.Sequential(*depthAttenBlock)
        self.featureAttention2 = selfAttention(squeezeFilters, squeezeFilters,3,1,1)
        self.featureAttention2_bn = nn.BatchNorm2d(squeezeFilters)
        
        # Output Layer
        self.convOut = nn.Conv2d(squeezeFilters,3,1,)

        # Weight Initialization
        self._initialize_weights()

    def forward(self, img):
        xInp = F.relu(self.inputConv_bn(self.inputConv(img)))
        xFA0 = F.relu(self.featureAttention0_bn(self.featureAttention0(xInp)))
        xGAP = self.globalPooling(xFA0)
        xSPE = self.spatialFeatExtBlock(xGAP) 
        xPUS = F.relu(self.psUpsampling(xSPE)) + xFA0
        xFA1 = F.relu(self.featureAttention1_bn(self.featureAttention1(xPUS)))
        XFFC = self.fullFeatCorelationBlock(xFA1)
        xFA2 =  F.relu(self.featureAttention2_bn(self.featureAttention2(XFFC))) + xFA1
        return torch.tanh(self.convOut(xFA2) + img)
    
    def _initialize_weights(self):
        self.inputConv.apply(init_weights)
        self.featureAttention0.apply(init_weights)
        self.globalPooling.apply(init_weights)
        self.spatialFeatExtBlock.apply(init_weights)
        self.psUpsampling.apply(init_weights)
        self.featureAttention1.apply(init_weights)
        self.fullFeatCorelationBlock.apply(init_weights)
        self.featureAttention2.apply(init_weights)
        self.convOut.apply(init_weights)

net = attentionNet(numAttentionBlock=8, scailingFactor=1)
summary(net, input_size = (3, 64, 64))
