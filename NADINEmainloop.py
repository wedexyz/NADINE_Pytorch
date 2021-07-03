import numpy as np
import time
import pdb
from utilsNADINE import meanStdCalculator, plotPerformance, labeledIdx
from NADINEbasic import NADINE
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import progressbar
import pdb
import warnings

def NADINEmain(NADINEnet,dataStreams,trainingBatchSize = 1, noOfEpoch = 1, labeled = True, nLabeled = 1):
    # import warnings
    # warnings.simplefilter('error', RuntimeWarning)

    # performance metrics
    # performanceMetrics = meanStd()   # [accuracy,testingLoss,testingTime,trainingTime]
    Accuracy     = []
    testing_Loss = []
    testingTime  = []
    trainingTime = []
    Y_pred       = []
    Y_true       = []
    Iter         = []
    
    accuracyHistory     = []
    lossHistory         = []
    hiddenNodeHistory   = []
    hiddenLayerHistory  = []
    winningLayerHistory = []

    # network evolution
    # netEvolution = meanStdCalculator()   # [nHiddenNode,nHiddenLayer]
    nHiddenNode  = []
    nHiddenLayer = []

    # batch loop
    bar = progressbar.ProgressBar(max_value=dataStreams.nBatch)
    for iBatch in range(0,dataStreams.nBatch):
        # load data
        batchIdx   = iBatch + 1
        batchData  = dataStreams.data[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        batchLabel = dataStreams.label[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        nBatchData = batchData.shape[0]

        # testing
        NADINEnet.testing(batchData,batchLabel)
        if iBatch > 0:
            Y_pred = Y_pred + NADINEnet.predictedLabel.tolist()
            Y_true = Y_true + NADINEnet.trueClassLabel.tolist()

            Accuracy.append(NADINEnet.accuracy)
            testing_Loss.append(NADINEnet.testingLoss)

        # if iBatch == 1 or iBatch%50 == 0:
            # print('\n')
            # print(iBatch,'- th batch of:', dataStreams.nBatch)
            # NADINEnet.dispPerformance()
        
        start_train = time.time()
        lblIdx = labeledIdx(nBatchData, nLabeled)
        # update dynamic learning rate
        NADINEnet.calculateAccuracyMatrices(batchLabel, lblIdx, labeled = labeled)
        NADINEnet.updateDynamicLr()

        if iBatch > 0:
            # drift detection
            NADINEnet.driftDetection()

            # grow layer
            NADINEnet.layerGrowing()

        # training data preparation
        if nLabeled < 1:
            NADINEnet.trainingDataPreparation(batchData[lblIdx],batchLabel[lblIdx])
        elif nLabeled == 1:
            NADINEnet.trainingDataPreparation(batchData,batchLabel)

        # training
        if NADINEnet.driftStatus == 0 or NADINEnet.driftStatus == 2:  # only train if it is stable or drift
            NADINEnet.training(batchSize = trainingBatchSize, epoch = noOfEpoch)

        end_train = time.time()
        training_time = end_train - start_train

        if iBatch > 0:
            # calculate performance
            testingTime.append(NADINEnet.testingTime)
            trainingTime.append(training_time)

            accuracyHistory.append(NADINEnet.accuracy)
            lossHistory.append(NADINEnet.testingLoss)
            
            # calculate network evolution
            nHiddenLayer.append(NADINEnet.nHiddenLayer)
            nHiddenNode.append(NADINEnet.nHiddenNode)

            hiddenNodeHistory.append(NADINEnet.nHiddenNode)
            hiddenLayerHistory.append(NADINEnet.nHiddenLayer)
            winningLayerHistory.append(NADINEnet.winLayerIdx+1)

            Iter.append(iBatch)

        bar.update(iBatch+1)
    
    
    print('\n')
    print('=== Performance result ===')
    print('Accuracy: ',np.mean(Accuracy),'(+/-)',np.std(Accuracy))
    print('Testing Loss: ',np.mean(testing_Loss),'(+/-)',np.std(testing_Loss))
    print('Precision: ',precision_score(Y_true, Y_pred, average='weighted'))
    print('Recall: ',recall_score(Y_true, Y_pred, average='weighted'))
    print('F1 score: ',f1_score(Y_true, Y_pred, average='weighted'))
    print('Testing Time: ',np.mean(testingTime),'(+/-)',np.std(testingTime))
    print('Training Time: ',np.mean(trainingTime),'(+/-)',np.std(trainingTime))

    print('\n')
    print('=== Average network evolution ===')
    print('Total hidden node: ',np.mean(nHiddenNode),'(+/-)',np.std(nHiddenNode))
    print('Number of layer: ',np.mean(nHiddenLayer),'(+/-)',np.std(nHiddenLayer))

    print('\n')
    print('=== Final network structure ===')
    NADINEnet.getNetProperties()
    
    # print('\n')f1_score
    # print('=== Precision Recall ===')
    # print(classification_report(Y_true, Y_pred))

    allPerformance = [np.mean(Accuracy),
                        f1_score(Y_true, Y_pred, average='weighted'),precision_score(Y_true, Y_pred, average='weighted'),
                        recall_score(Y_true, Y_pred, average='weighted'),
                        (np.mean(trainingTime)),np.mean(testingTime),
                        NADINEnet.nHiddenLayer,NADINEnet.nHiddenNode]

    performanceHistory = [Iter,accuracyHistory,lossHistory,hiddenNodeHistory,hiddenLayerHistory,winningLayerHistory]

    return NADINEnet, performanceHistory, allPerformance


def NADINEmainId(NADINEnet,dataStreams,trainingBatchSize = 1, noOfEpoch = 1, labeled = True, nLabeled = 1, nInitLabel = 1000):
    # import warnings
    # warnings.simplefilter('error', RuntimeWarning)

    # performance metrics
    # performanceMetrics = meanStd()   # [accuracy,testingLoss,testingTime,trainingTime]
    Accuracy     = []
    testing_Loss = []
    testingTime  = []
    trainingTime = []
    Y_pred       = []
    Y_true       = []
    Iter         = []
    
    accuracyHistory     = []
    lossHistory         = []
    hiddenNodeHistory   = []
    hiddenLayerHistory  = []
    winningLayerHistory = []

    # network evolution
    # netEvolution = meanStdCalculator()   # [nHiddenNode,nHiddenLayer]
    nHiddenNode  = []
    nHiddenLayer = []

    nInit = 0

    # batch loop
    bar = progressbar.ProgressBar(max_value=dataStreams.nBatch)
    for iBatch in range(0,dataStreams.nBatch):
        # load data
        batchIdx   = iBatch + 1
        batchData  = dataStreams.data[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        batchLabel = dataStreams.label[(batchIdx-1)*dataStreams.batchSize:batchIdx*dataStreams.batchSize]
        nBatchData = batchData.shape[0]

        nInit += nBatchData
        # testing
        NADINEnet.testing(batchData,batchLabel)
        if nInit > nInitLabel:
            if iBatch > 0:
                Y_pred = Y_pred + NADINEnet.predictedLabel.tolist()
                Y_true = Y_true + NADINEnet.trueClassLabel.tolist()

                Accuracy.append(NADINEnet.accuracy)
                testing_Loss.append(NADINEnet.testingLoss)

        # if iBatch == 1 or iBatch%50 == 0:
            # print('\n')
            # print(iBatch,'- th batch of:', dataStreams.nBatch)
            # NADINEnet.dispPerformance()
        
        start_train = time.time()
        if nInit <= nInitLabel:
            lblIdx = labeledIdx(nBatchData, nLabeled)
            # update dynamic learning rate
            NADINEnet.calculateAccuracyMatrices(batchLabel, lblIdx, labeled = labeled)
            NADINEnet.updateDynamicLr()

            if iBatch > 0:
                # drift detection
                NADINEnet.driftDetection()

                # grow layer
                NADINEnet.layerGrowing()

            # training data preparation
            if nLabeled < 1:
                NADINEnet.trainingDataPreparation(batchData[lblIdx],batchLabel[lblIdx])
            elif nLabeled == 1:
                NADINEnet.trainingDataPreparation(batchData,batchLabel)

            # training
            if NADINEnet.driftStatus == 0 or NADINEnet.driftStatus == 2:  # only train if it is stable or drift
                NADINEnet.training(batchSize = trainingBatchSize, epoch = noOfEpoch)

        end_train = time.time()
        training_time = end_train - start_train

        if nInit > nInitLabel:
            # calculate performance
            testingTime.append(NADINEnet.testingTime)
            trainingTime.append(training_time)

            accuracyHistory.append(NADINEnet.accuracy)
            lossHistory.append(NADINEnet.testingLoss)
            
            # calculate network evolution
            nHiddenLayer.append(NADINEnet.nHiddenLayer)
            nHiddenNode.append(NADINEnet.nHiddenNode)

            hiddenNodeHistory.append(NADINEnet.nHiddenNode)
            hiddenLayerHistory.append(NADINEnet.nHiddenLayer)
            winningLayerHistory.append(NADINEnet.winLayerIdx+1)

            Iter.append(iBatch)

        bar.update(iBatch+1)
    
    
    print('\n')
    print('=== Performance result ===')
    print('Accuracy: ',np.mean(Accuracy),'(+/-)',np.std(Accuracy))
    print('Testing Loss: ',np.mean(testing_Loss),'(+/-)',np.std(testing_Loss))
    print('Precision: ',precision_score(Y_true, Y_pred, average='weighted'))
    print('Recall: ',recall_score(Y_true, Y_pred, average='weighted'))
    print('F1 score: ',f1_score(Y_true, Y_pred, average='weighted'))
    print('Testing Time: ',np.mean(testingTime),'(+/-)',np.std(testingTime))
    print('Training Time: ',np.mean(trainingTime),'(+/-)',np.std(trainingTime))

    print('\n')
    print('=== Average network evolution ===')
    print('Total hidden node: ',np.mean(nHiddenNode),'(+/-)',np.std(nHiddenNode))
    print('Number of layer: ',np.mean(nHiddenLayer),'(+/-)',np.std(nHiddenLayer))

    print('\n')
    print('=== Final network structure ===')
    NADINEnet.getNetProperties()

    allPerformance = [np.mean(Accuracy),
                        f1_score(Y_true, Y_pred, average='weighted'),precision_score(Y_true, Y_pred, average='weighted'),
                        recall_score(Y_true, Y_pred, average='weighted'),
                        (np.mean(trainingTime)),np.mean(testingTime),
                        NADINEnet.nHiddenLayer,NADINEnet.nHiddenNode]
    
    # print('\n')f1_score
    # print('=== Precision Recall ===')
    # print(classification_report(Y_true, Y_pred))

    performanceHistory = [Iter,accuracyHistory,lossHistory,hiddenNodeHistory,hiddenLayerHistory,winningLayerHistory]

    return NADINEnet, performanceHistory, allPerformance