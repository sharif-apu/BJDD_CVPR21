import torch
import torch.nn as nn
from skimage import io, color
import numpy as np
from dataTools.dataNormalization import *
unNorm = UnNormalize()
class deltaEColorLoss(nn.Module):

    def __init__(self, normalize=None):
        super(deltaEColorLoss, self).__init__()
        self.loss = []
        self.normalize = normalize
        self.device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
    def torchTensorToNumpy(self, image):
        imageNP = unNorm(image).permute(1,2,0).cpu().detach().numpy() #.reshape(image.shape[1], image.shape[2], image.shape[0])
        return imageNP

    def __call__(self, genImage, gtImage):

        for pair in range(len(genImage)):
            
            # Converting and changing shape of torch tensor into numpy
            imageGTNP = self.torchTensorToNumpy(gtImage[pair])
            imageGenNP = self.torchTensorToNumpy(genImage[pair])

            # Calculating color difference
            deltaE = np.absolute(color.deltaE_ciede2000(color.rgb2lab(imageGTNP), color.rgb2lab(imageGenNP)))
            if self.normalize:
                deltaE /= 255.0

            # Mean deifference for an image pair
            self.loss.append(np.mean(deltaE))
        deltaELoss = torch.mean(torch.tensor(self.loss, requires_grad=True)).to(self.device)
        #print(deltaELoss)
        return deltaELoss
