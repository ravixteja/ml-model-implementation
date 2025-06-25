# Here we explore implementation of SVM
# using Radial Bsisi Function Kernel

# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load dataset
dataset = pd.read_csv('svm-kernel-dataset.csv')
# print(dataset.sample(5))

# plot for visualization
# plt.scatter(dataset['x1'],dataset['x2'])
# plt.show()
# we see the data is clearly not separable linearly

# separate based on class
pos_class = dataset[dataset['y']==1]
neg_class = dataset[dataset['y']==-1]

# plot for visualization of different classes
plt.scatter(pos_class['x1'],pos_class['x2'],marker='+')
plt.scatter(neg_class['x1'],neg_class['x2'],marker='.')
# plt.show()


# extract input and output features into numpy arrays
X = np.array(dataset.drop('y',axis=1))
Y = np.array(dataset.drop(['x1','x2'],axis=1))

# print(X)
# print(Y)

# define the kernel function
def rbfKernel(a,b,gamma):
    diff = a - b
    kernel_value = np.exp((-gamma) * (diff@diff.T))
    # print(kernel_value)
    return kernel_value

# rbfKernel(1.23,-8.73,0.1)

# now implement the minimization of dual objective function

# the dual objective function looks something like this
# Ld = sum_over_i(aplha_i) - 0.5 * sum_over_i,j(aplha_i*alpha_j*y_i*y_j*K(x_i,x_j))

# now we need to solve for -
# max_out_of_all_alpha_i>=0(Ld)
# subject to constraints
# 1. 0 < all alpha_i < C
# 2. sum_over_i(alpha_i*y_i) = 0

# solve using Quadratic Programming and the library cvxopt

# import the library
from cvxopt import matrix,solvers

# define regularization term
C = 5

# compute different parts of the dual

# compute kernel matrix - K(x_i,x_j)
# define gamma value
gamma = 0.1

# count=0
kernel_matrix = np.zeros((X.shape[0],X.shape[0]))
for i in range(X.shape[0]):
    # print(X[i])
    for j in range(X.shape[0]):
        # print(X[i]-X[j])
        val = rbfKernel(X[i],X[j],gamma)
        kernel_matrix[i,j] = val
        # count+=1
# print(count)
# print(kernel_matrix)

# compute other matrices

# here gram matrix would be the matrix computed by the kernel function * Y@Y.T
YX = matrix(kernel_matrix * np.dot(Y,Y.T))
# print(YX)

# to incorporate minus sign
minus = matrix(-1 * np.ones((X.shape[0],1)))
# print(minus)

# define constraints

# 1. 0 < all alpha_i < C

# c1_p1 = matrix(-1 * np.eye(X.shape[0]))
# c1_p2 = matrix(C * np.ones((X.shape[0],1))) # this sets only upper bound

# to set both upper and lower bounds
# stack the matrices
c1_p1 = matrix(np.vstack((
    -1 * np.eye(X.shape[0]),
    np.eye(X.shape[0])
    )))
c1_p2 = matrix(np.vstack((
    np.zeros((X.shape[0], 1)),
    C * np.ones((X.shape[0], 1))
    )))
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

sv_alphas_index = np.where(((alphas > 1e-5)&(alphas< C)).flatten())
# print(sv_alphas_index)
# print()
# print('alphas1')
# print(alphas[sv_alphas_index])
# print('alphas2')

# extract X and Y of support vectors
X_sv = X[sv_alphas_index]
Y_sv = Y[sv_alphas_index]
# print()
# print(X_sv)
# print(Y_sv)

# plot these points along with the dataset to visualize
# dataset
plt.scatter(pos_class['x1'],pos_class['x2'],marker='+',label='Class +1')
plt.scatter(neg_class['x1'],neg_class['x2'],marker='.',label='Class -1')
# support vectors
plt.scatter(X_sv[:,0],X_sv[:,1],facecolors='none', edgecolors='black', label='Support Vectors')
# plt.show()


# no weights matrix required

# calculate bias term

# wkt for the support vectors
# y(w.T*x + b) = 1
# so
# b = (1/y) - w.T*x
# calculate this for all support vectors and take average
bias = 0
for i in sv_alphas_index[0]:
    # print(X[i])
    # print(Y[i])
    summation = 0
    for j in range(X.shape[0]):
        summation = summation + alphas[j] * Y[j] * kernel_matrix[j, i]
    # print(summation)
    bias = bias + (Y[i]-summation)

bias = bias / len(sv_alphas_index[0])
print(bias)


# prediction function
def predictClass(test_point):
    result = 0
    for i in range(X.shape[0]):
        result = result + alphas[i] * Y[i] * rbfKernel(test_point,X[i],gamma)
    result = result + bias
    return 1 if result>0 else -1


test_point = [0,0]
test_point[0] = float(input('Enter X1:'))
test_point[1] = float(input('Enter X2:'))

prediction = predictClass(test_point)
if prediction > 0:
    label = 'Class +1'
else:
    label = 'Class -1'
print(f'The point{test_point} is classified as {label}')

# plot the test point for visualization
plt.scatter(test_point[0],test_point[1],label='Test Point',color='b')
plt.legend()
plt.xlabel('$X1$',fontsize=15)
plt.ylabel('$X2$',fontsize=15)
plt.title('Support Vector Machine Classifier using RBF Kernel',fontsize=15)
plt.savefig('svm-using-rbf-kernel.png')
plt.show()