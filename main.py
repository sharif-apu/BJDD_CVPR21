from utilities.parserUtils import *
from utilities.customUtils import *
from utilities.aestheticUtils import *
from dataTools.processDataset import *
from dataTools.patchExtractor import *
from mainModule.dpbs import *

if __name__ == "__main__":

    # Parsing Options
    options = mainParser(sys.argv[1:])
    if len(sys.argv) == 1:
        customPrint("Invalid option(s) selected! To get help, execute script with -h flag.")
        exit()
    
    # Reading Model Configuration
    if options.conf:
        configCreator()

    # Loading Configuration
    config = configReader()
  
    # Taking action as per received options
    if options.epoch:
        config=updateConfig(entity='epoch', value=options.epoch)
    if options.batch:
        config=updateConfig(entity='batchSize', value=options.batch)
    if options.manualUpdate:
        config=manualUpdateEntity()
    if options.modelSummary:
        DPBS(config).modelSummary()
    if options.train:
        DPBS(config).modelTraining(dataSamples=options.dataSamples)
    if options.retrain:
        DPBS(config).modelTraining(resumeTraning=True, dataSamples=options.dataSamples) 
    if options.inference:
        noiseSigmaSet = None
        if options.noiseSigma:
            noiseSigmaSet = options.noiseSigma.split(',')
            noiseSigmaSet = list(map(int, noiseSigmaSet))
        DPBS(config).modelInference(testImagesPath=options.sourceDir, outputDir=options.resultDir, noiseSet=noiseSigmaSet)
    if options.overFitTest:
        DPBS(config).modelTraining(overFitTest=True)
    if options.dataSampling:
        datasetSampler(config, options.sourceDir, options.resultDir, options.gridSize, options.dataSamples).samplingImages()
    if options.resumeDataSampling:
        datasetSampler(config, options.sourceDir, options.resultDir, options.gridSize, options.dataSamples).resumeSampling()
    if options.patch:
        patch = patchExtract(config, options.sourceDir, options.resultDir)
        patch()
    
        
        
        
            


