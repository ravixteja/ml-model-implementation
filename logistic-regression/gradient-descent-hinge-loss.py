import numpy as np

###################################################################
# Optimization problem
trainData = [
    #(x,y) pairs
    ((0,2),1),
    ((-2,0),1),
    ((1,-1),-1)
]

def phi(x):
    return np.array(x)

def initialWeightVector():
    return np.zeros(2)

def hingeLoss(w):
    return (1/len(trainData)*sum(max(1-w.dot(phi(x))*y,0) for x,y in trainData))

def gradientHingLoss(w):
    return (1/len(trainData)*sum(-phi(x)*y if 1-w.dot(phi(x))*y > 0 else 0 for x,y in trainData))

###################################################################
# Optimization algorithm
def gradientDescent(lossFunc, gradientLossFunc,initialWeights):
    w = initialWeights()
    lr = 0.1
    for x in range(500):
        costFunc = lossFunc(w)
        gradient = gradientLossFunc(w)
        w = w-(lr*gradient)
        print(f'epoch{x}: w: {w}, cost: {costFunc}, gradient: {gradient}')

###################################################################
# Using the Optimization algorithm
gradientDescent(hingeLoss, gradientHingLoss, initialWeightVector)