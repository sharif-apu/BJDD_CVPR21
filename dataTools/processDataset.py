import glob
import cv2
from pathlib import Path
import ntpath
import time
import argparse
import sys
from PIL import Image
import numpy as np
import os
from utilities.customUtils import *
from utilities.aestheticUtils import *
from etaprogress.progress import *
from dataTools.sampler import *


class datasetSampler:

    def __init__(self, config, source, target, gridSze, numberOfSamples=None):

        # Configuring parameters
        self.gtPath = formatDirPath(source)
        self.targetPath = formatDirPath(target)
        self.numberOfSamples = numberOfSamples
        print ("number of data samples to be processed",self.numberOfSamples)
        self.interval = int(config['interval'])
        self.barLen = int(config['barLen'])
        self.gridSze = int(gridSze)
        self.patchSize= 128

        # Creating a directory to save processed samples
        createDir(self.targetPath)
        
        # Listing all images stored in the source directory
        self.sourceDataSamples = imageList(formatDirPath(self.gtPath))

    def startResumeProcess(self):
        #self.numberOfSamples = numberOfSamples
        startTime = time.time()
        count = 0
        percent = 10
        printProgressBar(0, len(self.samplesInTargetDirectory), prefix = 'Loading process', suffix = 'completed', length = self.barLen)

        for s, i in enumerate(self.samplesInTargetDirectory):
            targetFile = self.gtPath + extractFileName(i)
            try:
                self.sourceDataSamples.remove(targetFile)
                
            except:
                pass
            printProgressBar(s, len(self.samplesInTargetDirectory), prefix = 'Loading process', suffix = ' completed', length = self.barLen)
            sys.stdout.flush()
        hours, minutes, seconds = timer(startTime, time.time())     
        print ("\nTime elapsed to resume process [{:0>2}:{:0>2}:{:0>2}]".format(hours, minutes, seconds))

        return self.sourceDataSamples

    def samplingImages (self):
        #self.numberOfSamples = numberOfSamples
        print ("in some fucking method",self.numberOfSamples)
        if not self.numberOfSamples:
            self.numberOfSamples = len(self.sourceDataSamples)
        i = 0
        timerFlag = 0 
        startTime = time.time()
        bar = ProgressBar(self.numberOfSamples, max_width=self.barLen)

        for sample in self.sourceDataSamples:
            # Tacking batch processing time
            if timerFlag == 0:
                loopTime = time.time()
                timerFlag = 1

            try:
                # Read Images
                patchSize = (self.patchSize, self.patchSize) 
                image =  Image.open(sample)
                image = np.asarray(image.resize(patchSize))

                # Sampling Data
                if self.gridSze == 1:
                    image = bayerSampler(image)
                if self.gridSze == 2:
                    image = quadBayerSampler(image)
                else:
                    image = dynamicBayerSampler(image,self.gridSze)
                image = Image.fromarray(image)
                image.save(self.targetPath+extractFileName(sample))
                i += 1

            except:
                os.remove(sample)
            if i % 1000 == 0:
                bar.numerator = i
                print("Image Sampled:", bar, end='\r')
            sys.stdout.flush()
            if i == self.numberOfSamples:
                print ("Successfully sampled target {} of images!".format(self.numberOfSamples))
                break
        hours, minutes, seconds = timer(startTime, time.time())     
        print ("Processed [{}] images! Total time elapsed [{:0>2}:{:0>2}:{:0>2}].".format(i,hours, minutes, seconds))


    def resumeSampling(self, numberOfSamples = None):
        #self.numberOfSamples = numberOfSamples
        # Resuming data processing  
        print ("Resuming Process....")
        self.samplesInTargetDirectory = imageList(formatDirPath(self.targetPath))
        if self.samplesInTargetDirectory:
            print ("[{}] image samples have been found in the target directory!".format(len(self.samplesInTargetDirectory)))
            if int(len(self.samplesInTargetDirectory)) >= int(len(self.sourceDataSamples)):
                print("All target images are already been processed! Thus, the process did not resume!")   
            else:
                if self.numberOfSamples:
                    if self.numberOfSamples - int(len(self.samplesInTargetDirectory)) > 0 and self.numberOfSamples < int(len(self.sourceDataSamples)):
                        self.numberOfSamples = self.numberOfSamples - int(len(self.samplesInTargetDirectory))

                
                    else:
                        print ("Invalid amount of target samples have been given!")
                        sys.exit()

                # Listing files already processed       
                self.sourceDataSamples = self.startResumeProcess()
                
                # Calling method to process samples
                self.samplingImages()
               
        else:
            print("Target directory is empty! Unable to resume process. Process is starting from the beggining...")
            self.samplingImages()
        
        

