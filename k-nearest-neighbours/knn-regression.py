# here we implement KNN Regression Algorithm from scratch

# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the dataset
data = pd.read_csv('dataset-for-regression.csv')

# convert to array
x = np.array(data.round(3))

# plot for visualization of data
# plt.scatter(data['x'],data['y'])
# plt.show()

# implementing the algorithm

# function to calculate squared distances
def calcSqdDistances(datapoint):
    distances = np.array((datapoint - x[:,0])**2)
    # print(distances)
    return distances

# function to arrange points in increasing order of their distances from a given point
def arrangePoints(datapoint):
    distances = calcSqdDistances(datapoint)
    dataFrame = data.copy()
    dataFrame['SqdDistances'] = distances
    # print(dataFrame)
    dataFrame = dataFrame.sort_values(by='SqdDistances',ascending=True)
    dataFrame = dataFrame.reset_index(drop=True)
    # print(dataFrame)
    return dataFrame

# the KNN Regression Algorithm
def KNNRegression(k,datapoint):
    KNearestPoints = arrangePoints(datapoint).head(k)
    # print(KNearestPoints)
    prediction = KNearestPoints['y'].mean()
    # print(prediction)
    return prediction

# calcSqdDistances(1.56)
# arrangePoints(1.56)
# KNNRegression(5,1.56)

# using the algorithm for predictions
datapoint = 7.98 # change the value here to predict for other points
prediction = (KNNRegression(5,datapoint)).round(3)
print('Predicted Value is ', prediction)

# plot for visualization
plt.scatter(x[:,0],x[:,1],marker='*',label='Datapoints')
plt.scatter(datapoint,prediction,color='r',label='Inference Point')
plt.legend()
plt.xlabel('$X$',fontsize=15)
plt.ylabel('$Y$',fontsize=15)
plt.show()