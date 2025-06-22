# Here, we explore implementation of Gaussian Naive Bayes Classification Algorithm from scratch

# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load dataset
# we are using a hypothetical dataset generated using python code
data = pd.read_csv('dataset-for-gaussian-nb-1.csv')
# there are two datasets in the directory,
# replace the csv file name to check for other dataset too

data = data.round(3)
x = np.array(data)

# plot for visualization of data
# plt.scatter(x[:,0],x[:,1])
# plt.show()

# separate based on classes
class_1 = data[data['y']==1]
class_0 = data[data['y']==0]

# plot for visualization
# plt.scatter(class_1['x1'],class_1['x2'],marker='^')
# plt.scatter(class_0['x1'],class_0['x2'],marker='*')
# plt.show()

# compute prior probabilities
prior_c1 = len(class_1)/len(data) 
prior_c0 = len(class_0)/len(data)
# print(prior_c1)
priors = np.array([prior_c0,prior_c1])

# compute mean and std deviation for each feature in each class
mean_c1 = np.array([])
std_c1 = np.array([])
mean_c0 = np.array([])
std_c0 = np.array([])

for i in range(np.array(class_1).shape[1] - 1):
    mean_c1 = np.append(mean_c1, [np.mean(np.array(class_1)[:,i]).round(3)])
    std_c1 = np.append(std_c1, [np.std(np.array(class_1)[:,i]).round(3)])
for i in range(np.array(class_0).shape[1] - 1):
    mean_c0 = np.append(mean_c0, [np.mean(np.array(class_0)[:,i]).round(3)])
    std_c0 = np.append(std_c0, [np.std(np.array(class_0)[:,i]).round(3)])

means = np.array([mean_c0,mean_c1])
stds = np.array([std_c0,std_c1])

# print(mean_c1,mean_c0)
# print(std_c1,std_c0)
# print(means)
# print(stds)
# print(means[0])

# define gaussian likelihood function
def gaussianLikelihood(dataarray, mean, std):
    part_a = 1/np.sqrt(2*np.pi*(std**2 + 1e-15))
    part_b = np.exp((-1/2*(std**2 + 1e-15)) * (dataarray - mean)**2)
    # print(part_a * part_b)
    return part_a * part_b

# gaussianLikelihood(np.array(class_1)[:,0],mean_c1[0],std_c1[0])

# define function to calculate scores for each class on a test point
def calculateScores(test_point):
    scores = np.array([])
    for i in range(len(priors)):
        class_score = np.log(priors[i]) + np.sum(np.log(gaussianLikelihood(test_point,means[i],stds[i])))
        # print(class_score)
        scores = np.append(scores, [class_score.round(3)])
    return scores

# calculateScores([1.4,2.5])

# define function to predict class
def predictClass(test_point):
    scores = calculateScores(test_point)
    # print(max(scores))
    # print(np.argmax(scores))
    return np.argmax(scores)

# Use the algorithm for prediction
datapoint = np.array([3.015,3.338])
prediction = predictClass(datapoint)
print(f'{datapoint} is classified as Class {prediction}')

# plot for visual intuition
plt.scatter(class_1['x1'],class_1['x2'],marker='^')
plt.scatter(class_0['x1'],class_0['x2'],marker='*')
plt.scatter(datapoint[0],datapoint[1],color='r',label='Test Point')
plt.legend()
plt.show()