# this is a test file 
# made to test step by step implementation of hard-margin-svm.py
# after it failed to model a proper discriminant function

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
dataset = pd.read_csv('svm-hard-margin-dataset.csv')
# incorporate bias term
dataset['bias_term'] = 1
bias_term = dataset.pop('bias_term')
dataset.insert(0,'bias_term',bias_term)

# print(dataset.sample(10))

# plot
# plt.scatter(dataset['x1'],dataset['x2'])
# plt.show()

# separate into classes
posi_class = dataset[dataset['y']==1]
nega_class = dataset[dataset['y']==-1]

# plot
# plt.scatter(posi_class['x1'],posi_class['x2'],marker='+')
# plt.scatter(nega_class['x1'],nega_class['x2'],marker='.')
# plt.show()

# extract features into input array X
X = np.array(dataset.drop('y',axis=1))
# print(X)

# extract labels into output array Y
Y = np.array(dataset.drop(['bias_term','x1','x2'],axis=1))

# implement algorithm

# can be implemented using two different ways - Gradient Descent and Quadratic Programming

# try gradient descent
# it must be run on the Hinge Loss function

# initialize weights
w = np.array([[0.89,0.34,-1.23]])

# define hinge loss
def hingeLoss(w):
    array_sum = 0
    for i in range(len(dataset)):
        array_sum += max(0, (1-(Y[i]*np.dot(w,X[i]))))
        # print(X[i])
        # print(Y[i])
    return array_sum

# total_loss = hingeLoss(dataset)
# print(total_loss)

# define gradient of hinge loss
def gradient(w):
    array_sum = 0
    for i in range(len(dataset)):
        if 1-(Y[i]*np.dot(w,X[i])) > 0:
            array_sum += -(Y[i]*X[i])
        # print(X[i])
        # print(Y[i])
    return array_sum

# gradient_val = gradient(dataset)
# print(gradient_val)

# apply the gradient descent algo
def gradientDescent(func, gradientFunc, initialWeights):
    w = initialWeights
    lr = 0.1
    for x in range(500):
        value = func(w)
        gradient = gradientFunc(w)
        w = w - lr * gradient
        # print(f'epoch{x}: weights = {w}, Cost: {value}, gradient: {gradient}')
    return w

w = gradientDescent(hingeLoss,gradient,w)
print(w)

# plot the decision boundary
# create range of values for x1 and x2
x1 = np.linspace(-1,3,50)
x2 = np.linspace(-2.5,3,50)

# implement the equation
x2 = -(w[0][0] + (w[0][1]*x1)) / w[0][2]
plt.scatter(posi_class['x1'],posi_class['x2'],marker='+')
plt.scatter(nega_class['x1'],nega_class['x2'],marker='.')
plt.plot(x1,x2,label='Final Decision Boundary',color='black')
plt.show()

# well the output shows it is classifiying all the points correctly
# but it it not the thing which we aim for in SVM
# in Hard Margin SVM decision boundary must be exactly in the centre.
# so I think Gradient Descent doesn't work well in case of SVM