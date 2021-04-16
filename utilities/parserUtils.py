import argparse
import sys

def mainParser(args=sys.argv[1:]):

    # Parser definition
    parser = argparse.ArgumentParser(description="Parses command.")

    # Parser Options
    parser.add_argument("-c", "--conf", action='store_true', help="Create/update config file")
    parser.add_argument("-ts", "--train", action='store_true', help="Start training with default parameters")
    parser.add_argument("-tr", "--retrain", action='store_true', help="Resume training with pretrained weights")
    parser.add_argument("-to", "--overFitTest", action='store_true', help="Over fitting testing")
    parser.add_argument("-e", "--epoch", type=int, help="Set number of epochs")
    parser.add_argument("-b", "--batch", type=int, help="Set batch size")
    parser.add_argument("-g", "--gridSize", type=int, default=2, help="Set batch size")
    parser.add_argument("-p", "--patch", action='store_true', help="Extract patches from target images")
    parser.add_argument("-i", "--inference", action='store_true', help="Inference with pretrained weights")
    parser.add_argument("-ns", "--noiseSigma", type=str, default=None, help="Set of sigma values to inference on pretrained weights")
    parser.add_argument("-s", "--sourceDir",  help="Directory to fetch images for testing")
    parser.add_argument("-d", "--resultDir",  help="Directory to save inference outputs")
    parser.add_argument("-ds", "--dataSampling", action='store_true', help="Create images samples for training")
    parser.add_argument("-dr", "--resumeDataSampling", action='store_true', help="Resume images sampling for training")
    parser.add_argument("-n", "--dataSamples", type=int, help="Specify number of images to be used for training/sampling")
    parser.add_argument("-u", "--manualUpdate", action='store_true', help="Manually update the configuration (entity)")
    parser.add_argument("-ms", "--modelSummary", action='store_true', help="Show the summary of models and configurations")
    
    options = parser.parse_args(args)

    #if options.inference and (options.sourceDir is None ):
    #    parser.error("--inference requires sourceDir ")

    if options.epoch and (not (options.train == True or options.retrain == True)):
        parser.error("--please enable training (-t) or retraining (-r) flag prior to update the number of epoch(s)")

    if options.batch and (not (options.train == True or options.retrain == True)):
        parser.error("--please enable training (-t) or retraining (-r) flag prior to set batch size")

    if options.dataSamples and (not (options.dataSampling == True or options.resumeDataSampling == True or options.train == True or options.retrain == True)):
        parser.error("--please enable data sampling (-ds) or resume data sampling (-dr) flag prior to set the number of target samples")
    
    if options.patch and (not (options.sourceDir and options.resultDir)):
        parser.error("--please provide the path to extract patches and saving directory!")
    return options