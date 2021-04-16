import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from modelDefinitions.basicBlocks import *    

class attentionNet(nn.Module):
    def __init__(self, squeezeFilters = 64, expandFilters = 64, depth = 3):
        super(attentionNet, self).__init__()

        # Input Block
        self.inputConv = nn.Conv2d(3, squeezeFilters, 3,1,1)
        depthAttenBlock = []
        for i in range (depth):
            depthAttenBlock.append(attentionGuidedResBlock(squeezeFilters, expandFilters))
        self.depthAttention1 = nn.Sequential(*depthAttenBlock)
        self.spatialAttention1 = SpatialAttentionBlock(squeezeFilters)
        self.down1 = nn.Conv2d(64, 128, 3, 2, 1) 

        depthAttenBlock1 = []
        for i in range (depth):
            depthAttenBlock1.append(attentionGuidedResBlock(128,128, dilationRate=1))
        self.depthAttention2 = nn.Sequential(*depthAttenBlock1)
        self.spatialAttention2 = SpatialAttentionBlock(128)
        self.down2 = nn.Conv2d(128, 256, 3, 2, 1) 

        depthAttenBlock3 = []
        for i in range (depth):
            depthAttenBlock3.append(attentionGuidedResBlock(256,256, dilationRate=1))
        self.depthAttention3 = nn.Sequential(*depthAttenBlock3)
        self.spatialAttention3 = SpatialAttentionBlock(256)
        self.convUP1 = nn.Conv2d(256, 128, 3, 1, 1) 
        self.psUpsampling1 = pixelShuffleUpsampling(inputFilters=128, scailingFactor=2)

        depthAttenBlock4 = []
        for i in range (depth):
            depthAttenBlock4.append(attentionGuidedResBlock(128,128, dilationRate=1))
        self.depthAttention4 = nn.Sequential(*depthAttenBlock4)
        self.spatialAttention4 = SpatialAttentionBlock(128)
        self.convUP2 = nn.Conv2d(128, 64, 3, 1, 1) 
        self.psUpsampling2 = pixelShuffleUpsampling(inputFilters=64, scailingFactor=2)


        # Output Block
        depthAttenBlock5 = []
        for i in range (depth):
            depthAttenBlock5.append(attentionGuidedResBlock(64,64, dilationRate=1))
        self.depthAttention5 = nn.Sequential(*depthAttenBlock5)
        self.spatialAttention5 = SpatialAttentionBlock(64)
        self.convOut = nn.Conv2d(squeezeFilters,3,1,)

        # Weight Initialization
        #self._initialize_weights()

    def forward(self, img):

        xInp = F.leaky_relu(self.inputConv(img))
        xSP1 = self.depthAttention1(xInp)
        xFA1 = F.leaky_relu(self.spatialAttention1(xSP1))

        xDS1 = F.leaky_relu(self.down1(xFA1))
        xSP2 = self.depthAttention2(xDS1)
        xFA2 = self.spatialAttention2(xSP2) 

        xDS2 = F.leaky_relu(self.down2(xFA2))
        xSP3 = self.depthAttention3(xDS2)
        xFA3 = self.spatialAttention3(xSP3)

        xCP1 = F.leaky_relu(self.convUP1(xFA3))
        xPS1 = self.psUpsampling1(xCP1) 
        xSP4 = self.depthAttention4(xPS1)
        xFA4 = self.spatialAttention4(xSP4) + xFA2

        xCP2 = F.leaky_relu(self.convUP2(xFA4))
        xPS2 = self.psUpsampling2(xCP2) 
        xSP5 = self.depthAttention5(xPS2)
        xFA5 = self.spatialAttention5(xSP5) + xFA1
        
        return torch.tanh(self.convOut(xFA5) + img)
        
        
    
    def _initialize_weights(self):

        self.inputConv.apply(init_weights)
        self.depthAttention1.apply(init_weights)
        self.spatialAttention1.apply(init_weights)
        
        self.down1.apply(init_weights)
        self.depthAttention2.apply(init_weights)
        self.spatialAttention2.apply(init_weights)
        
        self.down2.apply(init_weights)
        self.depthAttention3.apply(init_weights)
        self.spatialAttention3.apply(init_weights)
        
        self.convUP1.apply(init_weights)
        self.psUpsampling1.apply(init_weights)
        self.depthAttention4.apply(init_weights)
        self.spatialAttention4.apply(init_weights)
       
        self.convUP2.apply(init_weights)
        self.psUpsampling2.apply(init_weights)
        self.depthAttention5.apply(init_weights)
        self.spatialAttention5.apply(init_weights)
        
        self.convOut.apply(init_weights)

#net = attentionNet()
#summary(net, input_size = (3, 128, 128))
#print ("reconstruction network")