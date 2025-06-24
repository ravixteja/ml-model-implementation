# this is a test file 
# made to test step by step implementation of hard-margin-svm.py
# after it failed to model a proper discriminant function

# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from cvxopt import matrix, solvers

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

# extract input and output features into numpy arrays
X = np.array(dataset.drop('y',axis=1))
Y = np.array(dataset.drop(['bias_term','x1','x2'],axis=1))

# print(X)
# print(Y)

# now implement the minimization of dual objective function

# the dual objective function looks something like this
# Ld = sum_over_i(aplha_i) - 0.5 * sum_over_i,j(aplha_i*alpha_j*y_i*y_j*<x_i,x_j>)

# now we need to solve for -
# max_out_of_all_alpha_i>=0(Ld)
# subject to constraints
# 1. all alpha_i > 0
# 2. sum_over_i(alpha_i*y_i) = 0

# solve using Quadratic Programming and the library cvxopt

# import the library
from cvxopt import matrix,solvers

# compute different parts of the dual

# gram matrix - <x_i,x_j>
x_dot_prod_matrix = np.dot(X,X.T)
# print(x_dot_prod_matrix)

# y_i * y_j
y_dot_product_matrix = np.dot(Y,Y.T)
# print(y_dot_product_matrix)

# y_i*y_j*<x_i,x_j>
YX = matrix(x_dot_prod_matrix * y_dot_product_matrix)
# print(YX)

# to incorporate minus sign
minus = matrix(-1 * np.ones((X.shape[0],1)))
# print(minus)

# define constraints

# 1. all alpha_i > 0
c1_p1 = matrix(-1 * np.eye(X.shape[0]))
c1_p2 = matrix(np.zeros((X.shape[0],1)))
# print(c1_p1)
# print(c1_p2)

# 2. sum_over_i(alpha_i*y_i) = 0
c2_p1 = matrix(Y.reshape(1,-1).astype('double'))
c2_p2 = matrix(np.zeros((1,1)))
# print(c2_p1)
# print(c2_p2)

# solve the dual
solution = solvers.qp(YX,minus,c1_p1,c1_p2,c2_p1,c2_p2)

# fetch the alphas
alphas = np.array(solution['x'])
# print(alphas)

# extract support vectors - alphas >=0
# sv_alphas = alphas[(alphas > 1e-5).flatten()]
# print((alphas > 1e-5).flatten())

sv_alphas_index = np.where((alphas > 1e-5).flatten())
# print(sv_alphas_index)
# print()
# print(alphas[sv_alphas_index])

# extract X and Y of support vectors
X_sv = X[sv_alphas_index]
Y_sv = Y[sv_alphas_index]
# print()
# print(X_sv)
# print(Y_sv)

# plot these points along with the dataset to visualize
plt.scatter(pos_class['x1'],pos_class['x2'],marker='+')
plt.scatter(neg_class['x1'],neg_class['x2'],marker='.')
plt.scatter(X_sv[:,1],X_sv[:,2])
# plt.show()

# calculate the weights matrix
w = np.sum(alphas[sv_alphas_index] * X_sv * Y_sv, axis=0)
# print()
print(w)

# form the discriminant function(decision boundary)
x1 = np.linspace(-0.5,3.5,50)
x2 = np.linspace(-1,4,50)

x2 = -(w[0] + (w[1]*x1)) / w[2]

# plot the decision boundary
plt.plot(x1,x2,label='Final Decision Boundary',color='black')
plt.show()