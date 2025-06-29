# here we explore implementation of a simple Decision Tree classifier from scratch

# import libraries
import numpy as np
import pandas as pd

# load dataset
data = pd.read_csv('computer-purchase-data.csv')

# print(data.sample(5))

# we see that first 4 columns are input features and last one is the target variable

# define functions

# calculate gini impurity
def giniImpurity(dataset):
    # print(dataset.sample(10))
    mod_dataset = dataset.groupby(dataset['buys_computer'])
    # print(mod_dataset["age"].count()['n'])
    for i in range(dataset.shape[1]):
        print(mod_dataset)
    return None

giniImpurity(data)