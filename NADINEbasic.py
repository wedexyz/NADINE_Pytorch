import numpy as np
import time 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
import pdb
from collections import deque
import random
from scipy.stats.distributions import chi2
import pandas as pd
import warnings
from utilsNADINE import meanStdCalculator, probitFunc, deleteRowTensor, deleteColTensor
warnings.filterwarnings("ignore", category=RuntimeWarning)

class hiddenLayerBasicNet(nn.Module):
    def __init__(self, no_input, no_hidden):
        super(hiddenLayerBasicNet, self).__init__()
        # hidden layer
        self.linear = nn.Linear(no_input, no_hidden,  bias=True)
        self.activation = nn.Sigmoid()
        # self.activation = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        
        return x

class outputLayerBasicNet(nn.Module):
    def __init__(self, no_hidden, classes):
        super(outputLayerBasicNet, self).__init__()
        # softmax layer
        self.linearOutput = nn.Linear(no_hidden, classes,  bias=True)
        nn.init.xavier_uniform_(self.linearOutput.weight)
        self.linearOutput.bias.data.zero_()
        
    def forward(self, x):
        x = self.linearOutput(x)
        
        return x


class hiddenLayer():
    def __init__(self, no_input, no_hidden):
        self.network = hiddenLayerBasicNet(no_input,no_hidden)
        self.netUpdateProperties()

    def netUpdateProperties(self):
        self.nNetInput   = self.network.linear.in_features
        self.nNodes      = self.network.linear.out_features
        self.nParameters = (self.network.linear.in_features*self.network.linear.out_features +
                            len(self.network.linear.bias.data))

    def getNetProperties(self):
        print(self.network)
        print('No. of inputs :',self.nNetInput)
        print('No. of nodes :',self.nNodes)
        print('No. of parameters :',self.nParameters)

    def getNetParameters(self):
        print('Input weight: \n', self.network.linear.weight)
        print('Input bias: \n', self.network.linear.bias)

    def nodeGrowing(self,nNewNode = 1):
        nNewNodeCurr = self.nNodes + nNewNode
        
        # grow node
        # newWeight, newOutputWeight,_     = generateWeightXavInit(self.nNetInput,nNewNodeCurr,self.nOutputs,nNewNode)
        newWeight                        = nn.init.xavier_uniform_(torch.empty(nNewNode, self.nNetInput))
        self.network.linear.weight.data  = torch.cat((self.network.linear.weight.data,
                                                          newWeight),0)  # grow input weights
        self.network.linear.bias.data    = torch.cat((self.network.linear.bias.data,
                                                          torch.zeros(nNewNode)),0)  # grow input bias
        self.network.linear.out_features = nNewNodeCurr
        del self.network.linear.weight.grad
        del self.network.linear.bias.grad
        
        self.netUpdateProperties()

    def nodePruning(self,pruneIdx,nPrunedNode = 1):
        nNewNodeCurr = self.nNodes - nPrunedNode  # prune a node
        
        # prune node for current layer, output
        self.network.linear.weight.data  = deleteRowTensor(self.network.linear.weight.data,
                                                           pruneIdx)  # prune input weights
        self.network.linear.bias.data    = deleteRowTensor(self.network.linear.bias.data,
                                                           pruneIdx)  # prune input bias
        self.network.linear.out_features = nNewNodeCurr
        del self.network.linear.weight.grad
        del self.network.linear.bias.grad
        
        self.netUpdateProperties()

    def inputGrowing(self,nNewInput = 1):
        nNewInputCurr = self.nNetInput + nNewInput

        # grow input weight
        # _,_,newWeightNext = generateWeightXavInit(nNewInputCurr,self.nNodes,self.nOutputs,nNewInput)
        newWeightNext = nn.init.xavier_uniform_(torch.empty(self.nNodes, nNewInput))
        self.network.linear.weight.data = torch.cat((self.network.linear.weight.data,newWeightNext),1)
        del self.network.linear.weight.grad

        self.network.linear.in_features = nNewInputCurr
        self.netUpdateProperties()
        
    def inputPruning(self,pruneIdx,nPrunedNode = 1):
        nNewInputCurr = self.nNetInput - nPrunedNode

        # prune input weight of next layer
        self.network.linear.weight.data = deleteColTensor(self.network.linear.weight.data,pruneIdx)
        del self.network.linear.weight.grad

        # update input features
        self.network.linear.in_features = nNewInputCurr
        self.netUpdateProperties()


class outputLayer():
    def __init__(self, no_hidden, classes):
        self.network = outputLayerBasicNet(no_hidden,classes)
        self.netUpdateProperties()

    def netUpdateProperties(self):
        self.nNetInput   = self.network.linearOutput.in_features
        self.nOutputs    = self.network.linearOutput.out_features
        self.nParameters = (self.network.linearOutput.in_features*self.network.linearOutput.out_features +
                            len(self.network.linearOutput.bias.data))

    def getNetProperties(self):
        print(self.network)
        print('No. of inputs :',self.nNetInput)
        print('No. of output :',self.nOutputs)
        print('No. of parameters :',self.nParameters)

    def getNetParameters(self):
        print('Output weight: \n', self.network.linearOutput.weight)
        print('Output bias: \n', self.network.linearOutput.bias)

    def inputGrowing(self,nNewInput = 1):
        nNewInputCurr = self.nNetInput + nNewInput

        # grow input weight
        # _,_,newWeightNext = generateWeightXavInit(nNewInputCurr,self.nNodes,self.nOutputs,nNewInput)
        newWeightNext = nn.init.xavier_uniform_(torch.empty(self.nOutputs, nNewInput))
        self.network.linearOutput.weight.data = torch.cat((self.network.linearOutput.weight.data,newWeightNext),1)
        del self.network.linearOutput.weight.grad

        self.network.linearOutput.in_features = nNewInputCurr
        self.netUpdateProperties()
        
    def inputPruning(self,pruneIdx,nPrunedNode = 1):
        nNewInputCurr = self.nNetInput - nPrunedNode

        # prune input weight of next layer
        self.network.linearOutput.weight.data = deleteColTensor(self.network.linearOutput.weight.data,pruneIdx)
        del self.network.linearOutput.weight.grad

        # update input features
        self.network.linearOutput.in_features = nNewInputCurr
        self.netUpdateProperties()


class NADINE():
    def __init__(self,nInput,nOutput,alpha_w = 0.0005,alpha_d = 0.0001,LR = 0.02):
        # random seed control
        # np.random.seed(0)
        # torch.manual_seed(0)
        # random.seed(0)

        # initial network
        self.net = [hiddenLayer(nInput,nOutput),outputLayer(nOutput,nOutput)]

        # network significance
        self.averageBias  = meanStdCalculator()
        self.averageVar   = meanStdCalculator()
        self.averageInput = meanStdCalculator()

        # hyper parameters
        self.lr        = LR
        self.dynamicLr = [LR]   # dynamic learning rate for each hidden layer
        self.criterion = nn.CrossEntropyLoss()

        # drift detection parameters
        self.alphaWarning   = alpha_w
        self.alphaDrift     = alpha_d
        self.driftStatusOld = 0
        self.driftStatus    = 0
        self.driftHistory   = []
        
        # Evolving
        self.growNode   = False
        self.pruneNode  = False
        self.growLayer  = False
        self.pruneLayer = False

        # data
        self.bufferData        = torch.Tensor().float()
        self.bufferLabel       = torch.Tensor().long()
        self.accFmatrix        = deque([])
        self.anomalyDataNadine = anomalyDataDetector(nInput)
        self.sampleCategory    = torch.Tensor().long()   # 0: original samples; 1: anomaly samples; 2: augmented samples

        # properties
        self.nHiddenLayer = 1
        self.nHiddenNode  = nOutput
        self.nOutputs     = nOutput
        self.winLayerIdx  = 0

    def updateNetProperties(self):
        self.nHiddenLayer = len(self.net) - 1
        nHiddenNode = 0
        for iLayer in range(0,len(self.net)-1):
            nHiddenNode += self.net[iLayer].nNodes
        self.nHiddenNode = nHiddenNode

    def getNetProperties(self):
        for iLayer,nett in enumerate(self.net):
            print('\n',iLayer + 1,'-th layer')
            nett.getNetProperties()
        print('Dynamic laerning rate for each hidden layer: ',self.dynamicLr)

    # ============================= Evolving mechanism =============================
    def layerGrowing(self):
        if self.driftStatus == 2:
            nInput = self.net[-1].nNetInput
            
            del self.net[-1]
            
            self.net = self.net + [hiddenLayer(nInput,self.nOutputs),outputLayer(self.nOutputs,self.nOutputs)]

            self.dynamicLr.append(self.lr)
            
            self.averageBias = meanStdCalculator()
            self.averageVar  = meanStdCalculator()

            self.updateNetProperties()
            # self.winLayerIdentifier()
            # print('*** ADD a new LAYER ***')

    def hiddenNodeGrowing(self,layerIdx = -2):
        if layerIdx <= (len(self.net)-1):
            copyHiddenLayer = copy.deepcopy(self.net[layerIdx])
            copyHiddenLayer.nodeGrowing()
            self.net[layerIdx] = copy.deepcopy(copyHiddenLayer)

            if layerIdx == -2:
                # grow input for classifier
                copyOutputLayer = copy.deepcopy(self.net[layerIdx+1])
                copyOutputLayer.inputGrowing()
                self.net[-1] = copy.deepcopy(copyOutputLayer)
            else:
                copyNextNet = copy.deepcopy(self.net[layerIdx+1])
                copyNextNet.inputGrowing()
                self.net[layerIdx+1] = copy.deepcopy(copyNextNet)
        
            # print('+++ GROW a hidden NODE +++')
            self.updateNetProperties()
        else:
            raise IndexError
        
    def hiddenNodePruning(self,layerIdx = -2):
        if layerIdx <= (len(self.net)-1):
            copyHiddenLayer = copy.deepcopy(self.net[layerIdx])
            copyHiddenLayer.nodePruning(self.leastSignificantNode)
            self.net[layerIdx] = copy.deepcopy(copyHiddenLayer)

            if layerIdx == -2:
                # grow input for classifier
                copyOutputLayer = copy.deepcopy(self.net[layerIdx+1])
                copyOutputLayer.inputPruning(self.leastSignificantNode)
                self.net[-1] = copy.deepcopy(copyOutputLayer)
            else:
                copyNextNet = copy.deepcopy(self.net[layerIdx+1])
                copyNextNet.inputPruning(self.leastSignificantNode)
                self.net[layerIdx+1] = copy.deepcopy(copyNextNet)
        
            # print('+++ GROW a hidden NODE +++')
            self.updateNetProperties()
        else:
            raise IndexError

    # ============================= forward pass =============================
    def feedforwardTest(self,x,device = torch.device('cpu')):
        # feedforward to all layers
        with torch.no_grad():
            tempVar = x.to(device)
            tempVar = tempVar.type(torch.float)
            
            hList   = []

            for iLayer in range(len(self.net)):
                currnet = self.net[iLayer].network
                obj     = currnet.eval()
                obj     = obj.to(device)
                tempVar = obj(tempVar)
                if iLayer < len(self.net) - 1:
                    hList = hList + [tempVar.tolist()]

            self.hList                 = hList   # output of all hidden layers
            self.scoresTest            = tempVar
            self.multiClassProbability = F.softmax(tempVar.data,dim=1)
            self.rawOutput             = tempVar.data
            self.predictedLabelProbability, self.predictedLabel = torch.max(self.multiClassProbability, 1)

    def feedforwardTrain(self,x,device = torch.device('cpu')):
        # feedforward to the winning layer
        tempVar = x.to(device)
        tempVar = tempVar.type(torch.float)
        
        # feedforward to all layers
        for iLayer in range(len(self.net)):
            currnet = self.net[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(device)
            tempVar = obj(tempVar)

        self.scoresTrain = tempVar

    def feedforwardBiasVar(self,x,label_oneHot,device = torch.device('cpu')):
        # label_oneHot is label in one hot vector form, float, already put in device
        with torch.no_grad():
            tempVar = x.to(device)
            tempVar = tempVar.type(torch.float)
            
            hiddenNodeSignificance = []

            for iLayer in range(len(self.net)):
                currnet           = self.net[iLayer].network
                obj               = currnet.eval()
                obj               = obj.to(device)
                
                if iLayer == 0:
                    tempVar  = obj(tempVar)
                    tempVar2 = (tempVar.detach().clone())**2

                    # node significance
                    hiddenNodeSignificance.append(tempVar.detach().clone().squeeze(dim=0).tolist())

                else:
                    tempVar  = obj(tempVar)
                    tempVar2 = obj(tempVar2)

                    if iLayer < len(self.net) - 1:
                        # node significance 
                        hiddenNodeSignificance.append(tempVar.detach().clone().squeeze(dim=0).tolist())
                    
            # bias variance
            tempY    = F.softmax(tempVar,dim=1)                 # y
            tempY2   = F.softmax(tempVar2,dim=1)                # y2
            bias     = torch.norm((tempY - label_oneHot)**2)    # bias
            variance = torch.norm(tempY2 - tempY**2)            # variance

            self.bias     = bias.item()
            self.variance = variance.item()
            self.hiddenNodeSignificance = hiddenNodeSignificance

    # ============================= Network Evaluation =============================
    def calculateAccuracyMatrices(self, trueClassLabel, labeledDataIdx, labeled = True):
        # accuracy matrix for the whole network
        if labeled:
            self.F_matrix     = (self.predictedLabel != trueClassLabel).int().tolist()  # 1: wrong, 0: correct
        else:
            self.F_matrix = (self.predictedLabel[labeledDataIdx] != trueClassLabel[labeledDataIdx]).int().tolist()

    def driftDetection(self):
        # need to be modified
        self.driftStatusOld = self.driftStatus
        driftStatus = 0  # 0: no drift, 1: warning, 2: drift

        if np.max(self.F_matrix) != 0:
            
            # Prepare accuracy matrix.
            # combine buffer data, when previous batch is warning
            # F_matrix is the accuracy matrix of the current batch
            if self.driftStatusOld == 1:
                self.F_matrix = self.bufferF_matrix + self.F_matrix

            # combine current and previous feature matrix
            combinedAccMatrix = self.F_matrix

            # prepare statistical coefficient to confirm a cut point
            nData             = len(combinedAccMatrix)
            cutPointCandidate = [int(nData/4),int(nData/2),int(nData*3/4)]
            cutPoint          = 0
            errorBoundF       = np.sqrt((1/(2*nData))*np.log(1/self.alphaDrift))
            miu_F             = np.mean(self.F_matrix)   
            
            # confirm the cut point
            for iCut in cutPointCandidate:
                miu_E       = np.mean(combinedAccMatrix[0:iCut])
                nE          = len(combinedAccMatrix[0:iCut])
                errorBoundE = np.sqrt((1/(2*nE))*np.log(1/self.alphaDrift))
                if (miu_F + errorBoundF) <= (miu_E + errorBoundE):
                    cutPoint = iCut
                    # print('A cut point is detected cut: ', cutPoint)
                    break

            if cutPoint > 0:
                # prepare statistical coefficient to confirm a drift
                errorBoundDrift = ((np.max(combinedAccMatrix) - np.min(combinedAccMatrix))*
                                        np.sqrt(((nData - nE)/(2*nE*nData))*np.log(1/self.alphaDrift)))

                # if np.abs(miu_F - miu_E) >= errorBoundDrift:   # This formula is able to detect drift, even the performance improves
                if miu_E - miu_F >= errorBoundDrift:   # This formula is only able to detect drift when the performance decreses
                    # print('H0 is rejected with size: ', errorBoundDrift)
                    # print('Status: DRIFT')
                    driftStatus         = 2
                    self.accFmatrix     = deque([])
                    self.bufferF_matrix = []
                else:
                    # prepare statistical coefficient to confirm a warning
                    errorBoundWarning = ((np.max(combinedAccMatrix) - np.min(combinedAccMatrix))*
                                        np.sqrt(((nData - nE)/(2*nE*nData))*np.log(1/self.alphaWarning)))

                    # if np.abs(miu_F - miu_E) >= errorBoundWarning and self.driftStatusOld != 1:
                    if miu_E - miu_F >= errorBoundWarning and self.driftStatusOld != 1:
                        # print('H0 is rejected with size: ', errorBoundWarning)
                        # print('Status: WARNING')
                        driftStatus = 1
                        self.bufferF_matrix = self.F_matrix

                    else:
                        # print('H0 is NOT rejected')
                        # print('Status: STABLE')
                        driftStatus = 0
            else:
                # confirm stable
                # print('H0 is NOT rejected')
                # print('Status: STABLE')
                driftStatus = 0

        self.driftStatus = driftStatus
        self.driftHistory.append(driftStatus)

    def updateBiasVariance(self):
        # calculate mean of bias
        # should be executed after doing feedforwardBiasVar on the winning layer
        self.averageBias.updateMeanStd(self.bias)
        if self.averageBias.count < 1 or self.growNode:
            self.averageBias.resetMinMeanStd()
        else:
            self.averageBias.updateMeanStdMin()
        
        # calculate mean of variance
        self.averageVar.updateMeanStd(self.variance)
        if self.averageVar.count < 20 or self.pruneNode:
            self.averageVar.resetMinMeanStd()
        else:
            self.averageVar.updateMeanStdMin()

    def growNodeIdentification(self):
        dynamicKsigmaGrow = (1.25*np.exp(-self.bias) + 0.75) # (np.log(len(self.net)-1) + 1)
        growCondition1    = (self.averageBias.minMean + 
                             dynamicKsigmaGrow*self.averageBias.minStd)
        growCondition2    = self.averageBias.mean + self.averageBias.std

        if growCondition2 > growCondition1 and self.averageBias.count >= 1:
            self.growNode = True
        else:
            self.growNode = False
    
    def pruneNodeIdentification(self, layerIdx = -2):
        dynamicKsigmaPrune = (1.25*np.exp(-self.variance) + 0.75)
        pruneCondition1    = (self.averageVar.minMean + 
                              2*dynamicKsigmaPrune*self.averageVar.minStd)
        pruneCondition2    = self.averageVar.mean + self.averageVar.std
        
        if (pruneCondition2 > pruneCondition1 and not self.growNode and 
            self.averageVar.count >= 20 and
            self.net[layerIdx].nNodes > self.nOutputs):
            self.pruneNode = True
            self.findLeastSignificantNode()
        else:
            self.pruneNode = False

    def findLeastSignificantNode(self,layerIdx = -1):
        # find the least significant node in the winning layer
        # should be executed after doing feedforwardBiasVar on the winning layer
        self.leastSignificantNode = torch.argmin(torch.abs(torch.tensor(self.hiddenNodeSignificance[layerIdx]))).tolist()

    # def winLayerIdentifier(self):
    #     self.winLayerIdx = 0
    #     # idx = np.argmax(np.asarray(votWeight)/(np.asarray(allLoss) + 0.001))
    #     self.winLayerIdx = np.argmax(np.asarray(self.dynamicLr))

    # ============================= Training ============================= 
    def training(self,device = torch.device('cpu'),batchSize = 1,epoch = 1):
        # shuffle the data
        nData            = self.batchData.shape[0]
        
        # label for bias var calculation
        y_biasVar = F.one_hot(self.batchLabel, num_classes = self.net[-1].nOutputs).float()
        
        for iEpoch in range(0,epoch):

            shuffled_indices = torch.randperm(nData)

            for iData in range(0,nData,batchSize):
                # load data
                indices                  = shuffled_indices[iData:iData+batchSize]

                minibatch_xTrain         = self.batchData[indices]
                minibatch_xTrain         = minibatch_xTrain.to(device)
                minibatch_xTrain_biasVar = minibatch_xTrain

                minibatch_labelTrain     = self.batchLabel[indices]
                minibatch_labelTrain     = minibatch_labelTrain.to(device)
                minibatch_labelTrain     = minibatch_labelTrain.long()

                minibatch_sampleCategory = self.sampleCategory[indices]

                if iEpoch == 0:
                    minibatch_label_biasVar = y_biasVar[indices]
                    minibatch_label_biasVar = minibatch_label_biasVar.to(device)
                    
                    if batchSize > 1:
                        minibatch_xTrain_biasVar = torch.mean(minibatch_xTrain,dim=0).unsqueeze(dim=0)
                        minibatch_label_biasVar  = torch.mean(minibatch_label_biasVar,dim=0).unsqueeze(dim=0)

                    ## calculate mean of input
                    # self.averageInput.updateMeanStd(torch.mean(minibatch_xTrain,dim=0).unsqueeze(dim=0))
                    self.averageInput.updateMeanStd(minibatch_xTrain_biasVar)

                    ## get bias and variance
                    outProbit = probitFunc(self.averageInput.mean,self.averageInput.std)   # for Sigmoid activation function
                    self.feedforwardBiasVar(outProbit,minibatch_label_biasVar)             # for Sigmoid activation function
                    # self.feedforwardBiasVar(self.averageInput.mean,minibatch_label_biasVar)  # for ReLU activation function

                    # update bias variance
                    self.updateBiasVariance()

                    # growing
                    self.growNodeIdentification()
                    if self.growNode:
                        self.hiddenNodeGrowing()

                    # pruning
                    if not self.growNode:
                        self.pruneNodeIdentification()
                        if self.pruneNode:
                            self.hiddenNodePruning()

                # declare parameters to be trained
                optimizer = self.getTrainableParameters()

                # forward pass
                self.feedforwardTrain(minibatch_xTrain)
                loss = self.criterion(self.scoresTrain,minibatch_labelTrain)

                # backward pass
                optimizer.zero_grad()
                loss.backward()

                # apply gradient
                optimizer.step()

                if iEpoch == 0:                
                    # detect anomaly data
                    self.anomalyDataNadine.updateAnomaly(minibatch_xTrain,self.averageInput.mean,
                                                   self.scoresTrain.detach().clone(),minibatch_sampleCategory,indices)

            if iEpoch == 0:
                # add anomaly data to storage
                self.anomalyDataNadine.addDataToAnomaly(self.batchData,self.batchLabel,self.nHiddenLayer)
                

    def trainingDataPreparation(self, batchData, batchLabel, activeLearning = False,
        advSamplesGenrator = False, minorityClassList = None):

        if activeLearning:
            # sample selection
            # MCP: multiclass probability
            sortedMCP,_          = torch.sort(self.MultiClassProbability, descending=True)
            sortedMCP            = torch.transpose(sortedMCP, 1, 0)
            sampleConfidence     = sortedMCP[0]/torch.sum(sortedMCP[0:2], dim=0)
            indexSelectedSamples = sampleConfidence <= 0.75
            indexSelectedSamples = (indexSelectedSamples != 0).nonzero().squeeze()

            # selected samples
            batchData  = batchData[indexSelectedSamples]
            batchLabel = batchLabel[indexSelectedSamples]
            # print('selected sample size',batchData.shape[0])

        # training data preparation
        if self.driftStatus == 0 or self.driftStatus == 2:  # STABLE or DRIFT
            # check buffer
            if self.bufferData.shape[0] != 0:
                # add buffer to the current data batch
                self.batchData  = torch.cat((self.bufferData,batchData),0)
                self.batchLabel = torch.cat((self.bufferLabel,batchLabel),0)

                # clear buffer
                self.bufferData  = torch.Tensor().float()
                self.bufferLabel = torch.Tensor().long()
            else:
                # there is no buffer data
                self.batchData  = batchData
                self.batchLabel = batchLabel

            # provide data category for original samples
            # 0: original samples; 1: anomaly samples; 2: augmented samples
            nOriginalData       = self.batchData.shape[0]
            self.sampleCategory = torch.zeros(nOriginalData).long()

            if self.driftStatus == 2 and self.anomalyDataNadine.anomalyData.shape[0] != 0:
                # check anomaly data if drift
                # add anomaly data to the current data batch
                nAnomalyData    = self.anomalyDataNadine.anomalyData.shape[0]
                self.batchData  = torch.cat((self.anomalyDataNadine.anomalyData, self.batchData),0)
                self.batchLabel = torch.cat((self.anomalyDataNadine.anomalyLabel,self.batchLabel),0)
                # print('$$$ Anomaly data is added to the training set. Number of data: ',self.batchData.shape[0],'$$$')
                self.anomalyDataNadine.reset()

                # provide data category for anomaly data
                # 0: original samples; 1: anomaly samples; 2: augmented samples
                sampleCategoryAnomaly = torch.ones(nAnomalyData).long()
                self.sampleCategory   = torch.cat((self.sampleCategory,sampleCategoryAnomaly),0)

        if self.driftStatus == 1:  # WARNING
            # store data to buffer
            # print('Store data to buffer')
            self.bufferData  = batchData.detach().clone()
            self.bufferLabel = batchLabel.detach().clone()

        # generate adversarial samples for minority class
        if advSamplesGenrator and (self.driftStatus == 0 or self.driftStatus == 2):
            # prepare data
            if minorityClassList is not None and len(minorityClassList) != 0:

                nIdealData = int(self.batchData.shape[0]/self.net[-1].nOutputs)

                # select the minority class data
                # adversarialBatchData  = self.batchData [self.batchLabel == minorityClass]
                # adversarialBatchLabel = self.batchLabel[self.batchLabel == minorityClass]

                # nMinorityClass = adversarialBatchData.shape[0]
                # nMajorityClass = self.batchData.shape[0] - nMinorityClass

                nAdversarialSamples = 0

                for iClass in minorityClassList:
                    
                    if self.batchData [self.batchLabel == iClass].shape[0] == 0:
                        continue

                    # select the minority class data
                    adversarialBatchData  = self.batchData [self.batchLabel == iClass]
                    adversarialBatchLabel = self.batchLabel[self.batchLabel == iClass]

                    # forward pass
                    adversarialBatchData.requires_grad_()
                    self.feedforwardTrain(adversarialBatchData)
                    lossAdversarial = self.criterion(self.scoresTrain,adversarialBatchLabel)

                    # backward pass
                    lossAdversarial.backward()

                    nMinorityClass  = adversarialBatchData.shape[0]
                    nTimes          = int(nIdealData/nMinorityClass)
                    randConstSize   = adversarialBatchData.detach().clone().repeat(nTimes,1).shape[0]

                    adversarialData = (adversarialBatchData.detach().clone().repeat(nTimes,1) + 
                        0.01*torch.rand(randConstSize,1)*torch.sign(adversarialBatchData.grad).repeat(nTimes,1))
                    adversarialLabel = adversarialBatchLabel.repeat(nTimes)

                    # pdb.set_trace()
                    self.batchData  = torch.cat((self.batchData,adversarialData),0)
                    self.batchLabel = torch.cat((self.batchLabel,adversarialLabel),0)

                    nAdversarialSamples += adversarialData.shape[0]
                    # pdb.set_trace()

            else:
                # select all data
                adversarialBatchData  = self.batchData.detach().clone()
                adversarialBatchLabel = self.batchLabel.detach().clone()

                # forward pass
                adversarialBatchData.requires_grad_()
                self.feedforwardTrain(adversarialBatchData)
                lossAdversarial = self.criterion(self.scoresTrain,adversarialBatchLabel)

                # backward pass
                lossAdversarial.backward()

                # get adversarial samples
                adversarialBatchData = adversarialBatchData.detach().clone() + 0.007*torch.sign(adversarialBatchData.grad)

                self.batchData  = torch.cat((self.batchData,adversarialBatchData),0)
                self.batchLabel = torch.cat((self.batchLabel,adversarialBatchLabel),0)

                nAdversarialSamples = adversarialBatchData.shape[0]

            # provide data category for augmented data
            # 0: original samples; 1: anomaly samples; 2: augmented samples
            
            sampleCategoryAdversarial = 2*torch.ones(nAdversarialSamples).long()
            self.sampleCategory       = torch.cat((self.sampleCategory,sampleCategoryAdversarial),0)
            # print('selected sample size',self.batchData.shape[0])

    def updateDynamicLr(self):
        # calculate correlation between hidden node and output
        # use the correlation to update the dynamic learning rate for each layer
        if self.nHiddenLayer > 1 and self.driftStatus == 0:
            hrOutCorrCoeff = []
            y              = self.rawOutput.transpose(0,1)
            
            for i in range(len(self.hList)):
                currHr    = torch.FloatTensor(self.hList[i]).transpose(0,1)
                nCurrNode = torch.FloatTensor(self.hList[i]).transpose(0,1).shape[0]
                
                corrEachLayer = []
                
                for j in range(0,nCurrNode):
                    corrEachNode = []
                    for k in range(0,self.nOutputs):
                        currCorr = np.abs(np.corrcoef(currHr[j].tolist(),y[k].tolist())[0][1])
                        
                        if (currCorr != currCorr).any():
                            # print('There is NaN in calcDynamicLr')
                            # pdb.set_trace()
                            currCorr = 0.0001
                            
                        corrEachNode = corrEachNode + [currCorr]
                        
                    corrEachLayer = corrEachLayer + [np.average(corrEachNode)]
                    
                hrOutCorrCoeff = hrOutCorrCoeff + [np.average(corrEachLayer)]
                
            dLr = np.round(self.lr*np.exp(-1.0*(1.0/np.asarray(hrOutCorrCoeff) - 1.0)))
            dLr[dLr == 0.0] = 0.0001
            # print('adjust learning rate')

            self.dynamicLr = dLr.tolist()
            # self.winLayerIdentifier()

    def getTrainableParameters(self):
        # pdb.set_trace()
        for iLayer in range(len(self.net)):
            netOptim  = []
            netOptim  = netOptim + list(self.net[iLayer].network.parameters())
            if iLayer == 0:
                optimizer = torch.optim.SGD(netOptim, lr = self.dynamicLr[iLayer], momentum = 0.95)#,  weight_decay = 0.00005)
            elif iLayer > 0 and iLayer <= len(self.net) - 2:
                optimizer.add_param_group({'lr': self.dynamicLr[iLayer],'params': netOptim}) 
            else:
                optimizer.add_param_group({'lr': self.lr,'params': netOptim})
        return optimizer

    # ============================= Testing ==============================
    def testing(self,x,label,device = torch.device('cpu')):
        # load data
        x     = x.to(device)
        label = label.to(device)
        label = label.long()
        
        # testing
        start_test          = time.time()
        self.feedforwardTest(x)
        end_test            = time.time()
        self.testingTime    = end_test - start_test
        
        loss                = self.criterion(self.scoresTest,label)
        self.testingLoss    = loss.detach().item()
        correct             = (self.predictedLabel == label).sum().item()
        self.accuracy       = 100*correct/(self.predictedLabel == label).shape[0]  # 1: correct, 0: wrong
        self.trueClassLabel = label


class anomalyDataDetector(object):
    def __init__(self,nInput,minorityClass = None):
        self.nInput               = nInput
        self.Lambda               = 0.98               # Forgetting factor
        self.StabilizationPeriod  = 20                 # The length of stabilization period.
        self.indexStableExecution = nInput
        self.na                   = 10                 # number of consequent anomalies to be considered as change
        self.Threshold1           = chi2.ppf(0.95, df = nInput)
        self.Threshold2           = chi2.ppf(0.99,df = nInput)
        self.indexkAnomaly        = 0
        self.invCov               = torch.eye(nInput,nInput)
        self.center               = torch.zeros(1,nInput)
        self.caCounter            = 0
        self.anomalyData          = torch.Tensor().float()      # Identified anoamlies input
        self.anomalyLabel         = torch.Tensor().long()       # Identified anoamlies target
        self.anomalyIndices       = torch.Tensor().long()        # indices of Identified anoamlies target
        self.ChangePoints         = []                          # Index of identified change points
        
    def reset(self):
        self.indexkAnomaly  = 0
        self.invCov         = torch.eye(self.nInput,self.nInput)
        self.center         = torch.zeros(1,self.nInput)
        self.caCounter      = 0
        self.ChangePoints   = []
        self.anomalyIndices = torch.Tensor().long()
        
    def updateCenterCov(self,x):  
        # (InvCov,center,indexkAnomaly,Lambda,x)
        with torch.no_grad():
            default_Eff_Number = 200
            indexOfSample      = np.min([self.indexkAnomaly,default_Eff_Number])
            temp1              = self.mahalDist(x)
            temp1              = temp1 + (self.indexkAnomaly - 1)/self.Lambda
            multiplier         = ((self.indexkAnomaly)/((self.indexkAnomaly - 1)*self.Lambda))
            invCov             = (self.invCov - (torch.matmul(torch.matmul(self.invCov,(x - self.center).transpose(0,1)),
                                                              torch.matmul((x - self.center),self.invCov))/temp1))
            self.invCov        = multiplier*invCov
            self.center        = self.Lambda*self.center + (1.0 - self.Lambda)*x
        
    def updateAnomaly(self, x, averageInput, score, sampleCategory, indice, cnt = 1):
        for iData in range(len(x)):
            if sampleCategory[iData].item() == 0:   # only intended for original samples
                with torch.no_grad():
                    self.indexkAnomaly += cnt

                    if self.indexkAnomaly <= self.indexStableExecution:
                        self.center = averageInput

                    elif self.indexkAnomaly > self.indexStableExecution:
                        mahaldist        = self.mahalDist(x[iData:iData+1])
                        sortedScore,_    = torch.sort(F.softmax(score[iData:iData+1],dim=1),descending=True)
                        sortedScore      = sortedScore.squeeze(dim=0).tolist()
                        decisionBoundary = sortedScore[0]/(sortedScore[0] + sortedScore[1])

                        if self.indexkAnomaly > self.StabilizationPeriod:
                            # Threshold 1 and Threshold 2 are obtained using chi2inv
                            # (0.99,I) and chi2inv(0.999,I), the data point is regarded as an anomaly if
                            # the condition below is fulfilled. After this condition is
                            # executed, the CACounter is resetted to zero.
                            if ((mahaldist > self.Threshold1 and mahaldist <self.Threshold2) 
                                or decisionBoundary <= 0.55):
                                self.anomalyIndices = torch.cat((self.anomalyIndices,indice[iData:iData+1]),0)
                                self.caCounter = 0
                            else:
                                self.caCounter += cnt

                        if (self.caCounter >= self.na):
                            self.ChangePoints.append(self.indexkAnomaly - self.caCounter)
                            self.caCounter = 0

                        self.updateCenterCov(x[iData:iData+1])
    
    def addDataToAnomaly(self,data,label,nHiddenLayer):
        anomalyData         = torch.index_select(data,  0, self.anomalyIndices)
        anomalyLabel        = torch.index_select(label, 0, self.anomalyIndices)
        self.anomalyData    = torch.cat((self.anomalyData,anomalyData),0)
        self.anomalyLabel   = torch.cat((self.anomalyLabel,anomalyLabel),0)
        self.anomalyIndices = torch.Tensor().long()
        # print('selected sample size',self.anomalyData.shape[0])
        
        # if self.anomalyData.shape[0] > 5000:   # 5000*nHiddenLayer:
        if self.anomalyData.shape[0] > 2000*nHiddenLayer:
            # newIndex                 = self.anomalyData.shape[0] - 5000    #5000*nHiddenLayer - self.anomalyData.shape[0]
            newIndex                 = self.anomalyData.shape[0] - 2000*nHiddenLayer
            self.anomalyData         = self.anomalyData[newIndex:]
            self.anomalyLabel        = self.anomalyLabel[newIndex:]
            # print('selected sample size',self.anomalyData.shape[0])
            
    def mahalDist(self,x):
        with torch.no_grad():
            mahaldist = torch.matmul(torch.matmul((x-self.center),self.invCov),(x-self.center).transpose(0,1))
            self.mahaldist = mahaldist[0][0].tolist()
        
        return mahaldist