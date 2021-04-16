import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import sys
import glob
import time
import colorama
from colorama import Fore, Style
from etaprogress.progress import ProgressBar
from torchsummary import summary
from ptflops import get_model_complexity_info
from utilities.torchUtils import *
from dataTools.customDataloader import *
from utilities.aestheticUtils import *
from loss.pytorch_msssim import *
from loss.colorLoss import *
from loss.percetualLoss import *
from modelDefinitions.attentionDis import *
from modelDefinitions.attentionGen import *
from torchvision.utils import save_image


class DPBS:
    def __init__(self, config):
        
        # Model Configration 
        self.gtPath = config['gtPath']#"/media/sharif-apu/XtrasHD2/Dataset/test_large/"#
        self.targetPath = config['targetPath']#'dataset/sample/'#"/media/sharif-apu/XtrasHD2/sampledData/DecaHexa/"
        self.checkpointPath = "/media/sharif-apu/XtrasHD2/modelLog/checkpoint_ANETLight/"#config['checkpointPath']
        self.logPath = "/media/sharif-apu/XtrasHD2/modelLog/logDir_ANETLight/"#config['logPath']
        self.resultDir = config['resultDir']
        self.modelName = config['modelName']
        self.dataSamples = config['dataSamples']
        self.batchSize = int(config['batchSize'])
        self.imageH = 128#int(config['imageH'])
        self.imageW = 128#int(config['imageW'])
        self.inputC = int(config['inputC'])
        self.outputC = int(config['outputC'])
        self.scalingFactor = int(config['scalingFactor'])
        self.totalEpoch = int(config['epoch'])
        self.interval = int(config['interval'])
        self.learningRate = float(config['learningRate'])
        self.adamBeta1 = float(config['adamBeta1'])
        self.adamBeta2 = float(config['adamBeta2'])
        self.barLen = int(config['barLen'])
        
        # Initiating Training Parameters(for step)
        self.currentEpoch = 0
        self.startSteps = 0
        self.totalSteps = 0
        self.adversarialMean = 0
        self.PR = 0.25
        # Normalization
        self.unNorm = UnNormalize()
        

        # Preapring model(s) for GPU acceleration
        self.device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.attentionNet = attentionNet().to(self.device)
        self.discriminator = attentiomDiscriminator().to(self.device)

        # Optimizers
        self.optimizerEG = torch.optim.Adam(self.attentionNet.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
        self.optimizerED = torch.optim.Adam(self.discriminator.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
        
        # Scheduler for Super Convergance
        self.scheduleLR = None
        
    def customTrainLoader(self, overFitTest = False):
        
        targetImageList = imageList(self.targetPath)
        print ("Trining Samples (Input):", self.targetPath, len(targetImageList))

        if overFitTest == True:
            targetImageList = targetImageList[-1:]
        if self.dataSamples:
            targetImageList = targetImageList[:self.dataSamples]

        datasetReadder = customDatasetReader(   
                                                image_list=targetImageList, 
                                                imagePathGT=self.gtPath,
                                                height = self.imageH,
                                                width = self.imageW,
                                                scailingFactor = self.scalingFactor
                                            )

        self.trainLoader = torch.utils.data.DataLoader( dataset=datasetReadder,
                                                        batch_size=self.batchSize, 
                                                        shuffle=True
                                                        )
        
        return self.trainLoader

    def modelTraining(self, resumeTraning=False, overFitTest=False, dataSamples = None):
        
        if dataSamples:
            self.dataSamples = dataSamples 

        # Losses
        perceptualLoss = VGGPerceptualLoss(percepRegulator=1).to(self.device)
        L1Loss = torch.nn.L1Loss().to(self.device)
        ssimLoss = MSSSIM().to(self.device)
        colorLoss = deltaEColorLoss(normalize=True).to(self.device)
        adversarialLoss = nn.BCELoss().to(self.device)
 
        # Overfitting Testing
        if overFitTest == True:
            customPrint(Fore.RED + "Over Fitting Testing with an arbitary image!", self.barLen)
            trainingImageLoader = self.customTrainLoader(overFitTest=True)
            self.interval = 1
            self.totalEpoch = 100000
        else:  
            trainingImageLoader = self.customTrainLoader()


        # Resuming Training
        if resumeTraning == True:
            self.modelLoad()
            try:
                pass#self.modelLoad()

            except:
                #print()
                customPrint(Fore.RED + "Would you like to start training from sketch (default: Y): ", textWidth=self.barLen)
                userInput = input() or "Y"
                if not (userInput == "Y" or userInput == "y"):
                    exit()
        

        # Starting Training
        customPrint('Model training is about to begin using:' + Fore.YELLOW + '[{}]'.format(self.device).upper(), textWidth=self.barLen)
        
        # Initiating steps
        steps = len(trainingImageLoader)
        self.totalSteps =  int((steps*self.totalEpoch)/self.batchSize)
        startTime = time.time()
        
        # Instantiating Super Convergance 
        #self.scheduleLR = optim.lr_scheduler.OneCycleLR(optimizer=self.optimizerEG, max_lr=self.learningRate, total_steps=self.totalSteps)

        
        for currentStep in range(self.startSteps, self.totalSteps):

            # Initiating progress bar 
            bar = ProgressBar(self.totalSteps, max_width=int(self.barLen/2))

            # Time tracker
            iterTime = time.time()
            for LRImages, HRGTImages in trainingImageLoader:
                
                ##############################
                #### Initiating Variables ####
                ##############################

                # Updating Steps
                if currentStep > self.totalSteps:
                    self.savingWeights(currentStep)
                    customPrint(Fore.YELLOW + "Training Completed Successfully!", textWidth=self.barLen)
                    exit()
                currentStep += 1

                # Images
                rawInput = LRImages.to(self.device)
                highResReal = HRGTImages.to(self.device)
              
                
                # GAN Variables
                onesConst = torch.ones(rawInput.shape[0], 1).to(self.device)
                targetReal = (torch.rand(rawInput.shape[0],1) * 0.5 + 0.7).to(self.device)
                targetFake = (torch.rand(rawInput.shape[0],1) * 0.3).to(self.device)

                valid = Variable(Tensor(rawInput.shape[0], 1).fill_(1.0), requires_grad=False).to(self.device)
                fake = Variable(Tensor(rawInput.shape[0], 1).fill_(0.0), requires_grad=False).to(self.device)
                ##############################
                ####### Training Phase #######
                ##############################
    
                # Image Generation
                highResFake = self.attentionNet(rawInput)

                

                
                # Optimization of generator 
                self.optimizerEG.zero_grad()
                generatorContentLoss =  L1Loss(highResFake, highResReal) + \
                                        self.PR * perceptualLoss(highResFake, highResReal) + \
                                        colorLoss(highResFake, highResReal)

                #generatorAdversarialLoss = adversarialLoss(self.discriminator(highResFake), onesConst)

                pred_real = self.discriminator(highResReal.detach()) #discriminator(imgs_hr).detach()
                pred_fake = self.discriminator(highResFake)#discriminator(gen_hr)

                # Adversarial loss (relativistic average GAN)
                generatorAdversarialLoss = adversarialLoss(pred_fake - pred_real.mean(0, keepdim=True), valid)

                lossEG = generatorContentLoss * 5e-3 +  generatorAdversarialLoss * 1e-2 
                lossEG.backward()
                self.optimizerEG.step()


                pred_real = self.discriminator(highResReal)
                pred_fake = self.discriminator(highResFake.detach())

                # Optimaztion of Discriminator
                self.optimizerED.zero_grad()
                loss_real = adversarialLoss(pred_real - pred_fake.mean(0, keepdim=True), valid)
                loss_fake = adversarialLoss(pred_fake - pred_real.mean(0, keepdim=True), fake)

                # Total loss
                lossED = (loss_real + loss_fake) / 2
                #lossED = adversarialLoss(self.discriminator(highResReal), targetReal) + \
                #         adversarialLoss(self.discriminator(highResFake.detach()), targetFake)
                lossED.backward()
                self.optimizerED.step()
                # Steps for Super Convergance            
                #self.scheduleLR.step()

                ##########################
                ###### Model Logger ######
                ##########################   

                # Progress Bar
                if (currentStep  + 1) % self.interval/2 == 0:
                    bar.numerator = currentStep + 1
                    print(Fore.YELLOW + "Steps |",bar,Fore.YELLOW + "| LossEG: {:.4f}, LossED: {:.4f}, APR: {}".format(lossEG, lossED,self.PR),end='\r')
                    
                
                # Updating training log
                if (currentStep + 1) % self.interval == 0:
                   
                    # Updating Tensorboard
                    summaryInfo = { 
                                    'Input Images' : self.unNorm(rawInput),
                                    'AttentionNetGen Images' : self.unNorm(highResFake),
                                    'GT Images' : self.unNorm(highResReal),
                                    'Step' : currentStep + 1,
                                    'Epoch' : self.currentEpoch,
                                    'LossEG' : lossEG.item(),
                                    'LossED' : lossED.item(),
                                    'Path' : self.logPath,
                                    'Atttention Net' : self.attentionNet,
                                  }
                    tbLogWritter(summaryInfo)
                    save_image(self.unNorm(highResFake[0]), 'modelOutput.png')

                    # Saving Weights and state of the model for resume training 
                    self.savingWeights(currentStep)
                
                if (currentStep + 1) % (self.interval ** 2) == 0 : 
                    self.modelInference()
                    #if self.PR < 0.5: self.PR = round(self.PR + 0.10, 1)
                    eHours, eMinutes, eSeconds = timer(iterTime, time.time())
                    print (Fore.CYAN +'Steps [{}/{}] | Time elapsed [{:0>2}:{:0>2}:{:0>2}] | LossC: {:.2f}, LossP : {:.2f}, LossEG: {:.2f}, LossED: {:.2f}, SLR: {}' 
                            .format(currentStep + 1, self.totalSteps, eHours, eMinutes, eSeconds, colorLoss(highResFake, highResReal), perceptualLoss(highResFake, highResReal),lossEG, lossED, self.scheduleLR.get_last_lr()))
                    

        
    def modelInference(self, sourceDir = None, targetDir = None, imgW = None, imgH = None, gridSize = 4, multiCheckpoint = False):
        print("Inferancing about to begin.")
        if targetDir:
            self.resultDir = targetDir
        if imgH and imgW == None:
            imgW
        self.modelLoad()
        #self.attentionNet.eval()
        sourceDir = "/media/sharif-apu/XtrasHD/Testing Dataset/McM/"#self.targetPath#"testData/"
        print(sourceDir)
        testImageList = imageList(sourceDir)
        print (len(testImageList))
        #testImageList = testImageList[:100]
        with torch.no_grad():
            for imgPath in testImageList:
                #print(imgPath)
                img = inputForInference(imgPath).to(self.device)
                output = self.attentionNet(img)
                saveModelOutput(output, self.resultDir, imgPath)
        
    def multiInference(self):
        pass
    def modelSummary(self,input_size = None):
        if not input_size:
            input_size = (3, self.imageH//self.scalingFactor, self.imageW//self.scalingFactor)

     
        customPrint(Fore.YELLOW + "AttentionNet (Generator)", textWidth=self.barLen)
        summary(self.attentionNet, input_size =input_size)
        print ("*" * self.barLen)
        print()

        customPrint(Fore.YELLOW + "AttentionNet (Discriminator)", textWidth=self.barLen)
        summary(self.discriminator, input_size =input_size)
        print ("*" * self.barLen)
        print()

        flops, params = get_model_complexity_info(self.attentionNet, input_size, as_strings=True, print_per_layer_stat=False)
        customPrint('Computational complexity (Enhace-Gen):{}'.format(flops), self.barLen, '-')
        customPrint('Number of parameters (Enhace-Gen):{}'.format(params), self.barLen, '-')

        flops, params = get_model_complexity_info(self.discriminator, input_size, as_strings=True, print_per_layer_stat=False)
        customPrint('Computational complexity (Enhace-Dis):{}'.format(flops), self.barLen, '-')
        customPrint('Number of parameters (Enhace-Dis):{}'.format(params), self.barLen, '-')
        print()

        configShower()
        print ("*" * self.barLen)
    
    def savingWeights(self, currentStep):
        # Saving weights 
        checkpoint = { 
                        'step' : currentStep + 1,
                        'stateDictEG': self.attentionNet.state_dict(),
                        'stateDictED': self.discriminator.state_dict(),
                        'optimizerEG': self.optimizerEG.state_dict(),
                        'optimizerED': self.optimizerED.state_dict(),
                        'schedulerLR': self.scheduleLR
                        }
        saveCheckpoint(modelStates = checkpoint, path = self.checkpointPath, modelName = self.modelName)

    def modelLoad(self):

        customPrint(Fore.RED + "Loading pretrained weight", textWidth=self.barLen)

        previousWeight = loadCheckpoints(self.checkpointPath, self.modelName)
        self.attentionNet.load_state_dict(previousWeight['stateDictEG'])
        self.discriminator.load_state_dict(previousWeight['stateDictED'])
        self.optimizerEG.load_state_dict(previousWeight['optimizerEG']) 
        self.optimizerED.load_state_dict(previousWeight['optimizerED']) 
        self.scheduleLR = previousWeight['schedulerLR']
        self.startSteps = int(previousWeight['step'])
        
        customPrint(Fore.YELLOW + "Weight loaded successfully", textWidth=self.barLen)


        
