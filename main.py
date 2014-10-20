# import the necessary packages
from nolearn.dbn import DBN
import numpy as np
import pandas as pd
import pickle

# Output file
trainFile = "train.csv"
testFile = "test.csv"
modelFile = 'deeplearningModel'
outputFile = "output.csv"

# read in the training file
print "loading train data ..."
trainData = pd.DataFrame(pd.read_csv(trainFile, delimiter=","))
trainData = trainData.as_matrix()
trainLabels = trainData[:, 0]
trainFeats = trainData[:,1:] / 255.0

#set the training features
#read in the test file
print "loading test data ..."
testFeatures = pd.DataFrame(pd.read_csv(testFile, delimiter=","))
testFeatures = testFeatures.as_matrix() / 255.0

# train the Deep Belief Network with 784 input units (the flattened,
# 28x28 grayscale image), 300 hidden units, 10 output units (one for
# each possible output classification, which are the digits 1-10)
dbn = DBN(
    [trainFeats.shape[1], 800, 10],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=1)
# dbn.fit(trainFeats, trainLabels)

print "loading model ..."
filehandler = open(modelFile, 'r')
dbn = pickle.load(filehandler)
filehandler.close()

# print "saving model ..."
# filehandler = open(modelFile, 'w')
# pickle.dump(dbn, filehandler)
# filehandler.close()

# compute the predictions for the test data and show a classification
# report
print "making predictions ..."
predictedTest = dbn.predict(testFeatures)

# Generate submission
submission = {'ImageID': np.arange(1, predictedTest.shape[0]+1),
              'Label': predictedTest }
submission = pd.DataFrame(submission)
submission.to_csv(outputFile, index = 0)


