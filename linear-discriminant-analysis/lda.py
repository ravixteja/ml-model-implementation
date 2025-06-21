# here we implement Linear Discriminant Analysis from scratch

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load dataset
data = pd.read_csv('dataset.csv')

# extract input datapoints into array
X = np.array(data[['x1','x2']])

# plot the datapoints
# plt.scatter(X[:,0],X[:,1])
# plt.show()

# separate based on class labels
X1 = np.array(data[['x1','x2']][data['y']==1])
X0 = np.array(data[['x1','x2']][data['y']==0])

# compute mean for each class
class1_mean = np.array([np.mean(X1[:,0]).round(3),np.mean(X1[:,1]).round(3)])
class0_mean = np.array([np.mean(X0[:,0]).round(3),np.mean(X0[:,1]).round(3)])

# compute Within Class Variance
class1_var = (X1 - class1_mean).T @ (X1 - class1_mean)
class0_var = (X0 - class0_mean).T @ (X0 - class0_mean)
class_var = (class1_var + class0_var).round(3)
# print(class_var)

# compute between class variance
diff_of_means = (class1_mean - class0_mean).reshape(-1,1)
between_class_var = (diff_of_means @ diff_of_means.T).round(3)
# print(between_class_var)

# compute the eigen values and eigen vectors
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(class_var) @ between_class_var)
# print(eigvals)
# print(eigvecs)

# extract eigen vector with highest eigen value
lda_axis = eigvecs[:,np.argmax(eigvals)]
# print(lda_axis)

# convert to column vector
lda_axis = lda_axis.reshape(-1,1)
# print(lda_axis)

# project each datapoint onto the lda_axis
# print(X.shape)
# print(lda_axis.shape)
projections = X @ lda_axis # these are 1-d points

# for plotting we need 2-d points, so we convert to 2-d
lda_axis_unit = lda_axis / np.linalg.norm(lda_axis)
projections_2d = projections @ lda_axis_unit.T
# print(projections_2d)

# separating hyperplane

# compute projected means
proj_class1 = X1 @ lda_axis
proj_class0 = X0 @ lda_axis
proj_mean1 = np.mean(proj_class1)
proj_mean0 = np.mean(proj_class0)

# compute midpoint of projected means - threshold
threshold_1d = (proj_mean1 + proj_mean0)/2

# convert threshold to 2-d
threshold_2d = threshold_1d * lda_axis_unit
# print(threshold_2d)

# take direction perpendicular to lda_axis
perp_vec = np.array([-lda_axis_unit[1], lda_axis_unit[0]])

# generate points to plot
point1 = threshold_2d + 5 * perp_vec
point2 = threshold_2d - 5 * perp_vec


# plot for visual intuition

# original points
plt.scatter(X1[:,0],X1[:,1],marker='*',label='Class1')
plt.scatter(X0[:,0],X0[:,1],marker='.',label='Class0')

# class means
plt.scatter(class1_mean[0],class1_mean[1],label='Class1 Mean')
plt.scatter(class0_mean[0],class0_mean[1],label='Class0 Mean')

# projected points
plt.scatter(projections_2d[:,0],projections_2d[:,1],label='Projected Points',marker='+')

# lda axis
origin = np.mean(X, axis=0)
line_points = np.vstack([origin - 5 * lda_axis_unit.ravel(), origin + 5 * lda_axis_unit.ravel()])
plt.plot(line_points[:, 0], line_points[:, 1], color='brown', label='LDA Axis')

# projection lines
for i in range(len(X)):
    plt.plot([X[i, 0], projections_2d[i, 0]],
             [X[i, 1], projections_2d[i, 1]], 'gray', linestyle='--', linewidth=0.8)

# decision boundary
plt.plot([point1[0], point2[0]], [point1[1], point2[1]], linestyle='dashdot', label='Decision Boundary')

plt.title('Linear Discriminant Analysis',fontsize=15)
plt.xlabel('$X1$',fontsize=15)
plt.ylabel('$X2$',fontsize=15)
plt.legend()
plt.show()