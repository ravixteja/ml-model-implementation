# here we implement the perceptron learning algorithm from scratch
# Note: This only works well for perfectly linearly separable data.

# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load dataset
dataset = pd.read_csv('dataset-for-perceptron.csv')
# the dataset used here is perfectly linearly separable
# change the dataset here for training on other datapoints
dataset['bias_term'] = 1

bias_term = dataset.pop('bias_term')
dataset.insert(0,'bias_term',bias_term)

# plot for visualization
# plt.scatter(dataset['x1'],dataset['x2'])
# plt.show()

# print(dataset.sample(10))

# separate based on class labels - 1 and -1
pos_points = dataset[dataset['y']==1]
neg_points = dataset[dataset['y']==-1]

# plot to visualize separability
# plt.scatter(pos_points['x1'],pos_points['x2'],marker='+')
# plt.scatter(neg_points['x1'],neg_points['x2'],marker='.')
# plt.show()

# create an array copy of dataset, pos_points and neg_points
x = np.array(dataset.drop('y',axis=1))
x_pos_points = np.array(pos_points.drop('y',axis=1))
x_neg_points = np.array(neg_points.drop('y',axis=1))

# print(x)
# print(x.shape)

# initialize the weight matrix
# np.random.seed(125)
w = np.array(
    [
        [-0.0023],
        [0.079],
        [1.562]
    ]
)
# print(w)
# print(w.T.shape)

# now run the algorithm
epochs = 10
for i in range(epochs):
    misclassification_count = 0
    for i in range(x.shape[0]):
        y = dataset['y'][i]
        value = w.T @ x[i,:]

        # if product of y and w.x is negative - point is misclassified
        # update the weights
        if(y*value[0]<0):
            # print('misclassified',y,value)
            w = w + (x[i,:].reshape(-1,1)) * y
            misclassification_count +=1
            # print(w)
    print(f'Count of misclassified points = {misclassification_count}')
print(w)

# the decision boundary
# w0 + (w1*x1) + (w2*x2) = 0

# prediction for a new datapoint
datapoint = [0.9,0.75] # change the values here to predict for different datapoint
flag = 'Class -1'
if(w[0] + w[1]*datapoint[0] + w[2]*datapoint[1]>0):
    flag = 'Class 1'
print(flag)

# x2 = -(w0 + (w1*x1)) / w2

# create range of values for x1 and x2
x1 = np.linspace(-1,3,50)
x2 = np.linspace(-2.5,3,50)

# implement the equation
x2 = -(w[0] + (w[1]*x1)) / w[2]

# plot on graph with the datapoints
plt.scatter(pos_points['x1'],pos_points['x2'],marker='+',label='Class 1')
plt.scatter(neg_points['x1'],neg_points['x2'],marker='.',label='Class -1')
plt.scatter(datapoint[0],datapoint[1],label='Test Point',color='r')
plt.plot(x1,x2,label='Final Decision Boundary',color='black')
plt.legend()
plt.title('Perceptron Learning Algorithm Classifier',fontsize=15)
plt.xlabel('$x1$',fontsize=15)
plt.ylabel('$x2$',fontsize=15)
plt.show()