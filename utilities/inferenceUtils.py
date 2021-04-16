import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os 
import glob
from shutil import copyfile
import matplotlib.pyplot as plt
from utilities.customUtils import *
from dataTools.sampler import *
import numpy as np
import cv2
from PIL import Image
from dataTools.dataNormalization import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class AddGaussianNoise(object):
    def __init__(self, noiseLevel):
        self.var = 0.1
        self.mean = 0.0
        self.noiseLevel = noiseLevel
        
    def __call__(self, tensor):
        sigma = self.noiseLevel/100.
        noisyTensor = tensor + torch.randn(tensor.size()).uniform_(0, 1.) * sigma  + self.mean
        return noisyTensor 
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.var)


class inference():
    def __init__(self, gridSize, inputRootDir, outputRootDir, modelName, resize = None, validation = None ):
        self.inputRootDir = inputRootDir
        self.outputRootDir = outputRootDir
        self.gridSize = gridSize
        self.modelName = modelName
        self.resize = resize
        self.validation = validation
        self.unNormalize = UnNormalize()
    


    def inputForInference(self, imagePath, noiseLevel):
        img = Image.open(imagePath)
        #print(imagePath, img.size)
        #print (img.size)
        '''if  img.size[0]>1024:
            print("Image Resized",imagePath, img.size)
            resizeDimension =  (1024, img.size[1]) 
            img = img.resize(resizeDimension)
            print ("New Image Dimesion:", img.size)
            img.save(imagePath) 

        if  img.size[1]>1024:
            print("Image Resized",imagePath, img.size)
            resizeDimension =  (img.size[0], 1024) 
            img = img.resize(resizeDimension)
            print ("New Image Dimesion:", img.size)
            img.save(imagePath)''' 

        if  img.size[0]<600 or img.size[1]<600:
            #print("Image Resized",imagePath, img.size)
            resizeDimension =  (512, 512) 
            img = img.resize(resizeDimension)
            #print ("New Image Dimesion:", img.size)
            img.save(imagePath) 

        if  ("McM" in imagePath) or ("WED" in imagePath) or ("BSD" in imagePath):
            #print("Image Resized",imagePath, img.size)
            resizeDimension =  (512, 512) 
            img = img.resize(resizeDimension)
            img.save(imagePath) 
        if "Urban" in imagePath:
            #print("Image Resized",imagePath, img.size)
            resizeDimension =  (1024, 1024) 
            img = img.resize(resizeDimension)
            img.save(imagePath)  

        img = np.asarray(img) 
        if self.gridSize == 1 : 
            img = bayerSampler(img)
        elif self.gridSize == 2 : 
            img = quadBayerSampler(img)
        elif self.gridSize == 3 : 
            img = dynamicBayerSampler(img, gridSze)
        img = Image.fromarray(img)

        if self.resize:
            #resize(256,256)
            transform = transforms.Compose([ transforms.Resize(self.resize, interpolation=Image.BICUBIC) ])
            img = transform(img)

        transform = transforms.Compose([ transforms.ToTensor(),
                                        transforms.Normalize(normMean, normStd),
                                        AddGaussianNoise(noiseLevel=noiseLevel)])

        testImg = transform(img).unsqueeze(0)

        return testImg 


    def saveModelOutput(self, modelOutput, inputImagePath, noiseLevel, step = None, ext = ".png"):
        datasetName = inputImagePath.split("/")[-2]
        if step:
            imageSavingPath = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True)  + \
                            "_sigma_" + str(noiseLevel) + "_" + self.modelName + "_" + str(step) + ext
        else:
            imageSavingPath = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True)  + \
                            "_sigma_" + str(noiseLevel) + "_" + self.modelName + ext
        
        save_image(self.unNormalize(modelOutput[0]), imageSavingPath)

    

    def testingSetProcessor(self):

        testSets = glob.glob(self.inputRootDir+"*/")
        #print (testSets)
        if self.validation:
            #print(self.validation)
            testSets = testSets[:1]
        #print (testSets)
        testImageList = []
        for t in testSets:
            #print (t.split("/")[-2])
            testSetName = t.split("/")[-2]
            createDir(self.outputRootDir + self.modelName  + "/" + testSetName)
            imgInTargetDir = imageList(t, False)
            testImageList += imgInTargetDir
        return testImageList


