# Here we implement Hard Margin Support Vector Classifier from scratch

# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# load dataset
dataset = pd.read_csv('svm-hard-margin-dataset.csv')

# add column of 1's to incorporate bias term
dataset['bias_term'] = 1
bias_term = dataset.pop('bias_term')
dataset.insert(0,'bias_term',bias_term)

# print(dataset.sample(5))

# plot for visualization
# plt.scatter(dataset['x1'],dataset['x2'])
# plt.show()

# separate based on class
pos_class = dataset[dataset['y']==1]
neg_class = dataset[dataset['y']==-1]

# plot for visualization of different classes
# plt.scatter(pos_class['x1'],pos_class['x2'],marker='+')
# plt.scatter(neg_class['x1'],neg_class['x2'],marker='.')
# plt.show()

# convert dataset, pos_class, neg_class to np arrays of only inputs
x = np.array(dataset.drop('y',axis=1))
x_pos_class = np.array(pos_class.drop('y',axis=1))
x_neg_class = np.array(neg_class.drop('y',axis=1))
# print(x_neg_class)
# print((x@x.T).shape)
# print(np.dot(x,x.T) == x@x.T)

# extract np array of only outputs
y = np.array(dataset.drop(['bias_term','x1','x2'],axis=1))
# print(y)

# initialize weights
# w = np.array([
#     [0.25],
#     [-1.235],
#     [1.0458]
# ])

# now implement the algorithm

# compute Gram Matrix
K = np.dot(x,x.T) # this is the product - <xi xj> in the dual expression

# other computations in the Dual Objective function
Y = y @ y.T # this is the product - yi yj in the dual expression
P = matrix(K * Y) # this is the product - yi yj <xi xj> in the dual expression
# print(PQ.shape)
q = matrix(-1 * np.ones((x.shape[0],1)))
# print(q.shape)

# form the constraints

# constraint 1
G = matrix(-1 * np.eye(x.shape[0]))
# print(G.shape)
h = matrix(np.zeros((x.shape[0],1)))
# print(h.shape)

# constraint 2
A = matrix(y.reshape(1,-1).astype('double'))
b = matrix(np.zeros((1,1)))
# print(A.shape)
# print(b.shape)

# solve the dual constraint equation using Quadratic Programming
solution = solvers.qp(P,q,G,h,A,b)
alphas = np.array(solution['x'])
# print(alphas)

# extract support vectors
sv = (alphas > 1e-5).flatten()
sv_alphas = alphas[sv]
# print(sv_alphas)

sv_x = x[sv]
sv_y = y[sv]

print(sv_x,sv_y)

w = np.sum(sv_alphas * sv_y * sv_x, axis=0)
print(w)

# create range of values for x1 and x2
x1 = np.linspace(-1,3,50)
x2 = np.linspace(-2.5,3,50)

# implement the equation
x2 = -(w[0] + (w[1]*x1)) / w[2]
plt.scatter(pos_class['x1'],pos_class['x2'],marker='+')
plt.scatter(neg_class['x1'],neg_class['x2'],marker='.')
plt.plot(x1,x2,label='Final Decision Boundary',color='black')
plt.show()