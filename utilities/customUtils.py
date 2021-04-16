import ntpath
import time
import json
from pathlib import Path
import glob
import numpy 
import os, shutil
import colorama
from colorama import Fore, Style
from torchvision.utils import save_image
from colorama import init
from utilities.aestheticUtils import *
init(autoreset=True)

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)

def extractFileName(path, withoutExtension = None):
    ntpath.basename("a/b/c")
    head, tail = ntpath.split(path)

    if withoutExtension:
        return tail.split(".")[-2] or ntpath.basename(head).split(".")[-2]

    return tail or ntpath.basename(head)
    



def configCreator(path = 'mainModule/'):
    config = {}

    print( Fore.YELLOW + "Please configure following hyperparameters:")

    while(True):
        gtPath = formatDirPath(input("Path of ground gruth samples (i.e., /home/gtDir/): "))
        if not gtPath == "" :
            print(Fore.RED + "Ground truth path has been set to: ", gtPath)
            break

    while(True):
        targetPath = formatDirPath(input("Path of input images (i.e., /home/tarDir/): "))
        if not targetPath == "" :
            print(Fore.RED +  "Target path has been set to: ", targetPath)
            break
        
    checkpointPath = formatDirPath(input("Path to the checkpoint (default: checkpointDir/): ") or "checkpointDir/")
    logPath = formatDirPath(input("Path to the log files (default: logDir/): ") or "logDir/")
    testImagePath = formatDirPath(input("Path to save inference outputs(default: testImageDir/): ") or "testImageDir/")
    resultDir = formatDirPath(input("Path to save inference outputs(default: result/): ") or "logDir/")
    modelName = input("Name of model (default: DPBS): ") or "DBPS"
    dataSamples = input("Number of samples should be used for training/sampling (default: Undefined): ") or None
    interval = input("Number of steps to update log files (default: 100): ") or "100"
    batchSize = input("Batch size for model training (default: 16): ") or "16"
    barLen = input("Length of progress bar (default: 64): ") or "64"
    imageH = input("Height of input images (default: 256): ") or "256"
    imageW = input("Width of input images (default: 256): ") or "256"
    inputC = input("Number of input channels (default: 3): ") or "3"
    outputC = input("Number of output channels (default: 3): ") or "3"
    scalingFactor = input("Scaling factor for binning sensor (default: 4): ") or "4"
    binnigFactor = input("Binning factor of image sensor (default: 2): ") or "2"
    epoch = input("Number of total epochs (default: 50): ") or "50"
    learningRate = input("Learning rate (default: 0.0001): ") or "0.0001"
    adamBeta1 = input("Value of Adam Beta1 (default: 0.5): ") or "0.5"
    adamBeta2 = input("Value of Adam Beta2 (default: 0.99): ") or "0.99"

    # Updating dictionary
    config.update({ "gtPath": gtPath, 
                    "targetPath" : targetPath, 
                    "checkpointPath" : checkpointPath,
                    "testImagePath" : testImagePath,
                    "resultDir" : resultDir,
                    "logPath" : logPath, 
                    "modelName" : modelName,
                    "dataSamples" : dataSamples, 
                    "batchSize" : batchSize, 
                    "barLen" : barLen,
                    "interval" : interval,
                    "imageH" : imageH, 
                    "imageW" : imageW,
                    "inputC" : inputC,
                    "outputC" : outputC,
                    "scalingFactor" : scalingFactor,
                    "binnigFactor" : binnigFactor,
                    "epoch" : epoch, 
                    "learningRate" : learningRate, 
                    "adamBeta1" : adamBeta1, 
                    "adamBeta2" : adamBeta2})

    # Creating config file
    configWriter(config)
    return config

def manualUpdateEntity():
    
    while (True):
        entity = input("Enter name of key: ")
        value = input("Enter value for the corresponding key: ")
        config = updateConfig(entity, value) 

        userInput = input("Would you like to continue to update the config file? (default: N): ") or "N"
        if userInput == "N" or userInput == "n":
            break

    return config

def updateConfig(entity,value, path = 'mainModule/'):

    config = configReader()
    try:
        if config[entity] == value:
            print ("Noting to update!")
        else:
            print ("The value of config entity {} is changing from {} to {}".format(entity, config[entity], value))
            config[entity] = value
            userInput = input("Do you want to update config.json file? (default: N): ") or "N"
            if userInput == "Y" or userInput == "y":
                configWriter(config)
            else:
                print ("config.json file remainy unchanged!")
    
        return config
    except:
        print ("Incorrect input! Please refer to the following keys:")
        for key, value in config.items():
            print("Key name:", key)

        return None
            

def configReader(path = 'mainModule/'):

    try:
        with open(path+'config.json', 'r') as fp:
            config = json.load(fp)
    except:
        userInput = input("Unable to read config.json file! Would you like to create new config file? (default: N): ") or "N"
        if userInput == "Y" or userInput == "y":
            config = configCreator()
            print (config)
        else:
            print ("Process aborted! Please configure config.json file to continue!")
            exit()

    return config
            
   

def configWriter(config, path = 'mainModule/' ):
    with open(path+'config.json', 'w') as fp:
        json.dump(config, fp)
    print("Successfully updated config file!")
    return True

def createDir(path):
    # Create a directory to save processed samples
    Path(path).mkdir(parents=True, exist_ok=True)
    return True

def imageList(path, multiDir = False, imageExtension =['*.jpg', '*.png', '*.jpeg', '*.tif', '*.bmp']):
    #types = () # the tuple of file types
    imageList = []
    for ext in imageExtension:

        if multiDir == True:
            imageList.extend(glob.glob(path+"*/"+ext))
        else:
            imageList.extend(glob.glob(path+ext))
        
        imageList
    return imageList

def formatDirPath(path):

    if not path.endswith("/"):
        path = path.rstrip() + "/"

    return path

def removeFiles(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def configShower(textWidth = 64):

    config = configReader()
    customPrint(Fore.YELLOW + "Hyperparameters and Configurations", textWidth=textWidth)
    for c in config:
        customPrint("{}:".format(c).upper() + Fore.YELLOW + "{}".format(config[c]),textWidth=textWidth, style='-' )
   

