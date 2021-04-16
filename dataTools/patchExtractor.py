import glob
import cv2
import time
import numpy as np
from pathlib import Path
from colorama import Fore, Style
from etaprogress.progress import ProgressBar
from utilities.customUtils import *


class patchExtract:
    def __init__(self, config, sourceDir, targetPath):

        # Configuring parameters
        self.sourcePath = formatDirPath(sourceDir)
        self.targetPath = formatDirPath(targetPath)
        self.patchSize = int(config['imageW'])

        # Creating a directory to save processed samples
        createDir(self.targetPath)
        
        # Listing all images stored in the source directory
        self.sourceImages = imageList(self.sourcePath, True)

    def __call__(self):

        bar = ProgressBar(len(self.sourceImages), max_width=int(50))
        counter = 0
        for IC, i in enumerate(self.sourceImages):
            img = cv2.imread(i)
            imgTemp = img[:img.shape[0]- self.patchSize, : img.shape[1]- self.patchSize]
            for i in range(0, imgTemp.shape[0],  self.patchSize):
                for j in range(0, imgTemp.shape[1],  self.patchSize):
                    patch = img[i:i+ self.patchSize, j:j+ self.patchSize, :]
                    #print (patch.shape)
                    cv2.imwrite(self.targetPath+str(counter)+".png", patch)
                    counter += 1
            if IC % 2 == 0:
                bar.numerator = IC
                print(Fore.CYAN + "Image Processd |", bar,Fore.CYAN, end='\r')
            
        print ("\n Patch Extracted:", counter)