# here we explore implemantation of Logistic Regression from Scratch

####################################################################################
# Optimization Problem

# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data
data = pd.read_csv('dataset.csv')
data['1'] = 1
X = data[['1','x']].values
y = data['y'].values.reshape(-1,1)


# define the functions

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# cost function - Cross Entropy Loss Function
def costFunc(W):
    return -np.sum((y * np.log(sigmoid(X@W)+1e-15)) + ((1 - y)*np.log(1 - sigmoid(X@W)+1e-15)))

# gradients
def gradients(W):
    grad = (1/data.shape[0]) * np.dot(X.T, (sigmoid(X @ W) - y))  
    return grad

# initialize weights
def initializeWeights(X):
    return np.zeros(X.shape[1])

# predictor function
def predictClass(X_test,W):
    predictions = []
    for x in X_test:
        predictions.append(1 if sigmoid(x@W) > 0.5 else 0)
    return predictions

####################################################################################
# Optimization Algorithm
def gradientDescent(funcToOptimize, gradients, initialWeights):
    W = initialWeights(X).reshape(-1, 1)
    lr = 0.1
    for x in range(10000):
        value = funcToOptimize(W)
        grad = gradients(W)
        W = W - lr * grad
        # if x % 500 == 0:
        #     print(f'epoch{x}: w: {W.ravel()}, cost: {value:.4f}, gradient: {grad.ravel()}')
    return W

####################################################################################
# find the optimization
W = gradientDescent(costFunc,gradients,initializeWeights)

# make predictions
X_test = np.array([
    [1,2.29],
    [1,9.78],
    [1,5],
    [1,8.90]
])
predictions = predictClass(X_test,W)
print(predictions)


# plot for intuition
# scatter plot of datapoints
plt.scatter(data['x'],data['y'],label='datapoints',color='r',alpha=0.5)
# plot regression line
plt.plot(data['x'],X@W,label='Regression Line',color='b',alpha=0.5)
# plot sigmoid curve
plt.plot(data['x'],sigmoid(X@W),label='Sigmoid Curve',color='g',alpha=0.5)
# plot the point where probability = 0.5
plt.scatter((1/W[1]) * (-W[0]),0.5,label='Probability of 0.5',color='orange',alpha=0.5)

# plot the test values
plt.scatter(X_test[:,1],sigmoid(X_test@W),color='maroon',label='Test Points',marker='s')

plt.axvline(x=(1/W[1]) * (-W[0]), color='gray',linestyle='--')
plt.axhline(y=0.5, color='gray',linestyle='--')
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()
