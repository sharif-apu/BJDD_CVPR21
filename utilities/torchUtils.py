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
from dataTools.customTransform import *

un = UnNormalize()

def findLastWeights(path, modelName = None, lastWeight = None ):

    # Taking backup of last weights
    previousWeights = glob.glob(path+"*.pth")
    if modelName:
        path = path + modelName

    if len(previousWeights) > 1:
        try:
            weights = [int(extractFileName(p, True).split("_")[-1]) for p in previousWeights]
            weights.sort()
            lastWeight = path + "_checkpoint_{}.pth".format(weights[-1])
        except:
            print("Multi format checkpoints have been found! However, the checkpoint without epoch flag has been selected arbitarily.")
            lastWeight = path + "_checkpoint.pth"
    elif len(previousWeights) == 1:
        lastWeight = previousWeights[0]

    else:
        
        print ("Checkpoint directory is empty")

        return

    return lastWeight


def saveCheckpoint(modelStates, path, modelName = None, currentEpoch = None, backup = True):
    
    if modelName:
        cpPath = path+modelName
    else:
        cpPath = path
    createDir(path)

    if currentEpoch:
        cpName = cpPath + "_checkpoint_{}.pth".format(str(currentEpoch))
    else:
        cpName = cpPath + "_checkpoint.pth"
        
    
    if backup:
        # Taking backup of last weight
        backupPath = path+ "/backup/"
        createDir(backupPath)
        removeFiles(backupPath)
        
        if (len(glob.glob(path+"*.pth")) < 1):
            pass
        elif (len(glob.glob(path+"*.pth")) > 1):
            lastWeight = findLastWeights(path, modelName)
            copyfile(lastWeight, backupPath+extractFileName(lastWeight))
        else:
            copyfile(cpName, backupPath+extractFileName(cpName))

    torch.save(modelStates, cpName )




def loadCheckpoints(path, modelName, epoch=False, lastWeights = True):

    # Checking wights saving format
    if lastWeights == True:
        cpPath = findLastWeights(path, modelName)
    else:
        cpPath = path + modelName

    # Loading checkpoint
    checkpoint = torch.load(cpPath)
    return checkpoint

def loadCheckpointsGAN(generator, discriminator, optimizerG, optimizerD, path, modelName, epoch=False, lastWeights = True):

    # Checking wights saving format
    if lastWeights == True:
        cpPath = findLastWeights(path, modelName)
    else:
        cpPath = path + modelName

    # Loading checkpoint
    checkpoint = torch.load(cpPath)
    generator.load_state_dict(checkpoint['stateDictG'])
    discriminator.load_state_dict(checkpoint['stateDictD'])
    optimizerG.load_state_dict(checkpoint['optimizerG'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])
    epoch = checkpoint['epoch']
    print("Previous weights loaded successfully!")

    return generator, discriminator, optimizerG, optimizerD, epoch



def show_img(img):
    plt.figure(figsize=(18,15))
    # unnormalize
    img = img / 2 + 0.5  
    npimg = img.numpy()
    npimg = np.clip(npimg, 0., 1.)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def tbLogWritter2(model, loss, currentSteps, epoch, inputImage, outputImage, gtImage, path):
    
    createDir(path)
    # Defining summary writter
    writer = SummaryWriter(path + 'epoch_{}'.format(epoch))

    # Writiting images to the tensorboard
    writer.add_scalar('Training Loss', loss, currentSteps)
    writer.add_image('Input images', torchvision.utils.make_grid(inputImage))
    writer.add_image('Output images', torchvision.utils.make_grid(outputImage))
    writer.add_image('GT images', torchvision.utils.make_grid(gtImage))
    writer.add_graph(model, inputImage)

    writer.close()

def tbLogWritter(summaryInfo):
    
    createDir(summaryInfo['Path'])
    writer = SummaryWriter(summaryInfo['Path'] + 'epoch_{}'.format(summaryInfo['Epoch']))

    # Defining summary writter
    for k in summaryInfo:
        #print (k)
        if 'Image' in k:
            writer.add_image(k, torchvision.utils.make_grid(summaryInfo[k]), summaryInfo['Step'])

        elif 'Loss' in k:
            #print(k)
            writer.add_scalar(k, summaryInfo[k])

        elif 'Model' in k:
            writer.add_graph(summaryInfo[k], summaryInfo['Input Image'])

    writer.close()

def inputForInference(path, imgW = 256, imgH = 256, gridSize = 4):

    img = Image.open(path)
    #img = img.resize((2048,1280))
    #print (path)
    #img = quadBayerSampler(img)
    #if imgH and imgW:
    #    img = cv2.resize(img,(imgW,imgH))
    
    img = np.asarray(img) 
    img = quadBayerSampler(img)
    img = Image.fromarray(img)
    transformHRGT = transforms.Compose([  
                                            #transforms.Resize((256,256), interpolation=Image.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize(normMean, normStd),
                                            transforms.RandomApply([AddGaussianNoise(0., .1)], p=1)
                                            ])

    imgTest = transformHRGT(img).unsqueeze(0)
    #print (imgTest.max(), imgTest.min())
    save_image(un(imgTest), "ÍnputImage.png")

    return imgTest #* 255


def saveModelOutput(output, path, fileName, ext = ".png"):
   
    createDir(path)
    
    imageSavingPath = path + extractFileName(fileName, True) +"_ANET"+ ext
    #output = un(output) 
    #outputImg = output.reshape(output.shape[2],output.shape[3],3).squeeze(0).cpu().numpy()
    #print (outputImg.shape, type(outputImg))
    '''finalImage = output[0].reshape(output.shape[2],output.shape[3],3) #* 255
    finalImage = finalImage.cpu().numpy()
    print(finalImage.shape, finalImage.max(), finalImage.min())
    #print (finalImage)
    cv2.imwrite(  imageSavingPath, finalImage )
    #finalImage.save(path + 'pilsave.png')'''
    save_image(un(output[0]), imageSavingPath)
    #print("Image seccessfully saved!")

#lastWeight = findLastWeights("checkpointDir/")
#print(lastWeight)