import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from modelDefinitions.basicBlocks import *    

class attentionNet(nn.Module):
    def __init__(self, squeezeFilters = 32, expandFilters = 64, scailingFactor = 2, numAttentionBlock=10):
        super(attentionNet, self).__init__()

        # Input Layer
        self.inputConv = nn.Conv2d(3, squeezeFilters, 3,1,1)
        self.inputConv_bn = nn.BatchNorm2d(squeezeFilters)

        # Feature Attention 1
        self.featureAttention0 = selfAttention(squeezeFilters, squeezeFilters,3,1,1)
        self.featureAttention0_bn = nn.BatchNorm2d(squeezeFilters)
        
        # Spatial Feature Extraction
        self.globalPooling =  nn.AvgPool2d(2,2) 
        self.ab1 = attentionGuidedResBlock(squeezeFilters, expandFilters)
        self.ab2 = attentionGuidedResBlock(squeezeFilters, expandFilters)
        self.ab3 = attentionGuidedResBlock(squeezeFilters, expandFilters)
        self.ab4 = attentionGuidedResBlock(squeezeFilters, expandFilters)
        self.ab5 = attentionGuidedResBlock(squeezeFilters, expandFilters)
        self.ab6 = attentionGuidedResBlock(squeezeFilters, expandFilters)
        self.ab7 = attentionGuidedResBlock(squeezeFilters, expandFilters)
        self.ab8 = attentionGuidedResBlock(squeezeFilters, expandFilters)
        self.ab9 = attentionGuidedResBlock(squeezeFilters, expandFilters)
        self.ab10 = attentionGuidedResBlock(squeezeFilters, expandFilters)
        self.ab11 = attentionGuidedResBlock(squeezeFilters, expandFilters)
        self.ab12 = attentionGuidedResBlock(squeezeFilters, expandFilters)
        
        # Feature Attention 2
        self.featureAttention1 = selfAttention(squeezeFilters, squeezeFilters,3,1,1)
        self.featureAttention1_bn = nn.BatchNorm2d(squeezeFilters)

        # Pixel shuffle Upscailing
        self.psUpsampling = pixelShuffleUpsampling(inputFilters=squeezeFilters, scailingFactor=2)

        # Feature Attention 3
        self.featureAttention2 = selfAttention(squeezeFilters, squeezeFilters,3,1,1)
        self.featureAttention2_bn = nn.BatchNorm2d(squeezeFilters)
       
        # Full feature corelation
        self.fb1 = attentionGuidedResBlock(squeezeFilters, expandFilters,  bn=False)
        self.fb2 = attentionGuidedResBlock(squeezeFilters, expandFilters,  bn=False)
        self.fb3 = attentionGuidedResBlock(squeezeFilters, expandFilters,  bn=False)
        self.fb4 = attentionGuidedResBlock(squeezeFilters, expandFilters,  bn=False)
        self.fb5 = attentionGuidedResBlock(squeezeFilters, expandFilters,  bn=False)
        self.fb6 = attentionGuidedResBlock(squeezeFilters, expandFilters,  bn=False)

        # Feature Attention 4
        self.featureAttention3 = selfAttention(squeezeFilters, squeezeFilters,3,1,1)
        self.featureAttention3_bn = nn.BatchNorm2d(squeezeFilters)

        # Output Layer
        self.convOut = nn.Conv2d(squeezeFilters,3,1,)

    

        # Weight Initialization
        self._initialize_weights()

    def forward(self, img):
        xInp  = F.relu(self.inputConv_bn(self.inputConv(img)))
        xFA0  = F.relu(self.featureAttention0_bn(self.featureAttention0(xInp)))
        xGAP  = self.globalPooling(xFA0)
        xAB1  = self.ab1(xGAP)
        xAB2  = self.ab2(xAB1) + xAB1
        xAB3  = self.ab3(xAB2) + xAB2
        xAB4  = self.ab4(xAB3) + xAB3
        xAB5  = self.ab5(xAB4) + xAB4
        xAB6  = self.ab6(xAB5) + xAB5
        xAB7  = self.ab7(xAB6) + xAB6
        xAB8  = self.ab8(xAB7) + xAB7
        xAB9  = self.ab9(xAB8) + xAB8
        xAB10 = self.ab10(xAB9)  + xAB9
        #xAB11 = self.ab11(xAB10) + xAB10
        #xAB12 = self.ab11(xAB11) + xAB11
        
        xFA1  = F.relu(self.featureAttention1_bn(self.featureAttention1(xAB10)))
        xUPS  = F.relu(self.psUpsampling(xFA1)) + xInp#+ xFA0

        xFA2  = F.relu(self.featureAttention2_bn(self.featureAttention2(xUPS)))
        xFB1  = self.fb1(xFA2)
        xFB2  = self.fb2(xFB1) + xFB1
        xFB3  = self.fb2(xFB2) + xFB2 
        xFB4  = self.fb2(xFB3) + xFB3 
        xFB5  = self.fb2(xFB4) + xFB4
        #xFB6  = self.fb2(xFB4) + xFB5
        xFA3  = F.relu(self.featureAttention3_bn(self.featureAttention3(xFB5))) + xFA2

        return torch.tanh(self.convOut(xFA3) + img)
    
    def _initialize_weights(self):
        self.inputConv.apply(init_weights)
        self.featureAttention0.apply(init_weights)
        self.globalPooling.apply(init_weights)
        self.ab1.apply(init_weights)
        self.ab2.apply(init_weights)
        self.ab3.apply(init_weights)
        self.ab4.apply(init_weights)
        self.ab5.apply(init_weights)
        self.ab6.apply(init_weights)
        self.ab7.apply(init_weights)
        self.ab8.apply(init_weights)
        self.ab9.apply(init_weights)
        self.ab10.apply(init_weights)
        self.ab11.apply(init_weights)
        self.ab12.apply(init_weights)
        self.featureAttention1.apply(init_weights)
        self.psUpsampling.apply(init_weights)
        self.featureAttention2.apply(init_weights)
        self.fb1.apply(init_weights)
        self.fb2.apply(init_weights)
        self.fb3.apply(init_weights)
        self.fb4.apply(init_weights)
        self.fb5.apply(init_weights)
        self.fb6.apply(init_weights)
        self.featureAttention3.apply(init_weights)
        self.convOut.apply(init_weights)

#net = attentionNet(numAttentionBlock=8, scailingFactor=1)
#summary(net, input_size = (3, 64, 64))
