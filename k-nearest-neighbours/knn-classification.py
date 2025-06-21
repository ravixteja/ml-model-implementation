# here we implement KNN Classification Algorithm from scratch

# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the dataset
data = pd.read_csv('dataset-for-classification.csv')

# convert to array
x = np.array(data)

# plot for visualization of data
# plt.scatter(data['x1'],data['x2'])
# plt.show()

# separate based on class
x1 = np.array(data[data['y']==1])
x0 = np.array(data[data['y']==0])

# plotting again
# plt.scatter(x1[:,0],x1[:,1],marker='*')
# plt.scatter(x0[:,0],x0[:,1],marker='+')
# plt.show()

# designing the algorithm

# function to calculate squared distances from a given point
def calcSqdDistances(datapoint):
    distances = ((datapoint[0] - x[:,0])**2 + (datapoint[1] - x[:,1])**2).round(3)
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

# the KNN Classification Algorithm
def KNNClassification(k, datapoint):
    KNearestPoints = arrangePoints(datapoint).head(k)
    count = KNearestPoints.groupby(['y']).count()
    # print(count)
    prediction = count['x1'].idxmax()
    # print(prediction)
    return prediction

# using the algorithm for predictions
datapoint = [3.456,3.587] # change the values here to predict for other points
prediction = KNNClassification(5,datapoint)
print('Predicted Class is ', prediction)

# plot for visualization
plt.scatter(x1[:,0],x1[:,1],marker='*',label='Class1')
plt.scatter(x0[:,0],x0[:,1],marker='+',label='Class0')
plt.scatter(datapoint[0],datapoint[1],color='r',label='Inference Point')
plt.legend()
plt.xlabel('$X1$',fontsize=15)
plt.ylabel('$X2$',fontsize=15)
plt.show()