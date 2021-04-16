import torch
import cv2
from torchvision.utils import save_image
import random

class AddGaussianNoise(object):
    def __init__(self, mean=0., var=.1, pov = 0.6):
        self.var = var
        self.mean = mean
        self.pov = pov
    def __call__(self, tensor):
        sigma = random.uniform(0, self.var ** self.pov)
        noisyTensor = tensor + torch.randn(tensor.size()).uniform_(0, 1.) * sigma  + self.mean
        return noisyTensor 
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.var)

'''if __name__ == "__main__":
    img = cv2.imread("dataTools/1.png")#/255.0
    print(img.shape, img.max(), img.min())
    tmgTr = torch.tensor(img)
    noise = AddGaussianNoise()
    imgNoiseTorch = noise(tmgTr)
    #print (imgNoiseTorch.shape)
    cv2.imwrite("dataTools/noiseTest.png", imgNoiseTorch.numpy())#*255)'''
    