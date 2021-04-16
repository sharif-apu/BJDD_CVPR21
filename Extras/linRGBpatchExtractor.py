import glob
import cv2
import time
import numpy as np
from pathlib import Path
from colorama import Fore, Style
from etaprogress.progress import ProgressBar
#from utilities.customUtils import *
import ntpath

def quadBayerSampler(image):
    img  = image.copy()
    
    # Quad R 
    img[::4,::4, 1:3]=0
    img[1::4,1::4, 1:3]=0
    img[::4,1::4, 1:3]=0
    img[1::4,::4, 1:3]=0

    # Quad B 
    img[3::4,2::4, 0:2]=0
    img[3::4,3::4, 0:2]=0
    img[2::4,3::4, 0:2]=0
    img[2::4,2::4, 0:2]=0

    #Quad G12
    img[1::4,2::4, ::2]=0
    img[1::4,3::4, ::2]=0
    img[::4,2::4, ::2]=0
    img[::4,3::4, ::2]=0
    
    #Quad G21
    img[2::4,1::4, ::2]=0
    img[3::4,1::4, ::2]=0
    img[2::4,::4, ::2]=0
    img[3::4,::4, ::2]=0


    return img

def createDir(path):
    # Create a directory to save processed samples
    Path(path).mkdir(parents=True, exist_ok=True)
    return True

def imageList(path, multiDir = False, imageExtension =['*.jpg', '*.png', '*.jpeg']):
    #types = () # the tuple of file types
    imageList = []
    for ext in imageExtension:

        if multiDir == True:
            imageList.extend(glob.glob(path+"*/"+ext))
        else:
            imageList.extend(glob.glob(path+ext))
        
        imageList
    return imageList

def extractFileName(path, withoutExtension = None):
    ntpath.basename("a/b/c")
    head, tail = ntpath.split(path)

    if withoutExtension:
        return tail.split(".")[-2] or ntpath.basename(head).split(".")[-2]

    return tail or ntpath.basename(head)
    

def formatDirPath(path):

    if not path.endswith("/"):
        path = path.rstrip() + "/"

    return path

class patchExtract:
    def __init__(self,  sourcePath, targetPath):

        # Configuring parameters
        self.sourcePath = formatDirPath(sourcePath)
        self.targetPath = formatDirPath(targetPath)
        self.pathGTPatch = "gtPatch/"
        self.pathQBPatch = "quadBayerPatch/"
        self.patchSize = 256#int(config['imageW'])

        # Creating a directory to save processed samples
        createDir(self.pathGTPatch)
        createDir(self.pathQBPatch)
        
        # Listing all images stored in the source directory
        self.sourceImages = imageList(self.sourcePath)
        print (len(self.sourceImages))


    def __call__(self):

        bar = ProgressBar(len(self.sourceImages), max_width=int(50))
        counter = 0
        for IC, i in enumerate(self.sourceImages):
            #print(i)
            tarImgPath = extractFileName(i)
            #print(self.targetPath+tarImgPath)
            img = cv2.imread(i)
            linImg = cv2.imread(self.targetPath+tarImgPath)
            #print(img.shape)
            imgTemp = img[:img.shape[0]- self.patchSize, : img.shape[1]- self.patchSize]
            for i in range(0, imgTemp.shape[0],  self.patchSize):
                for j in range(0, imgTemp.shape[1],  self.patchSize):
                    patch = img[i:i+ self.patchSize, j:j+ self.patchSize, :]
                    LinRGB = linImg[i:i+ self.patchSize, j:j+ self.patchSize, :]
                    sampledLinRGB = quadBayerSampler(LinRGB)
                    #print (patch.shape)
                    cv2.imwrite(self.pathGTPatch+str(counter)+".png", patch)
                    #cv2.imwrite(self.pathGTPatch+str(counter)+"_lRGB.png", LinRGB)
                    cv2.imwrite(self.pathQBPatch+str(counter)+".png", sampledLinRGB)
                    counter += 1
            if IC % 2 == 0:
                bar.numerator = IC
                print(Fore.CYAN + "Image Processd |", bar,Fore.CYAN, end='\r')
            
        print ("\n Patch Extracted:", counter)


sourcePath = "DIV2K_train_HR"
targetPath = "linRGB"

p = patchExtract(sourcePath, targetPath)
p()