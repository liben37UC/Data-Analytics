from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#This example is taken from Machine Learning with Python by William Gray


#loading iris data this function is loaded as a numpy array with shape 150x4 rowsxcolumns
data=load_iris().data

#numpy array with size 150 rows
labels=load_iris().target

#to concatenate the two numpy arrays we need to reshape the labels array
labels=np.reshape(labels,(150,1))

#now we can concatenate the two arrays: labels and data axis=0 concats by row and axis=1 concats by column and axis -1 concats based on second dimension
data=np.concatenate([data,labels],axis=-1)

#names will be correlated with how our matrix is arranged 4 columns for attributes and 1 label for species
names=['sepal-length','sepal-width','petal-length','petal-width','species']

#import the data using pandas
dataset=pd.DataFrame(data,columns=names)

#to see what the first 5 rows looks like use head() for end use tail
#print(dataset.tail(5))

plt.figure(1, figsize=(10,8))
plt.scatter(data[:50,0], data[:50,1], c="r", label="Iris-setosa")
plt.scatter(data[50:100,0], data[50:100,1], c="g", label="Iris-versicolor")
plt.scatter(data[100:,0], data[100:,1], c="b", label="Iris-virginica")
plt.xlabel('Sepal length', fontsize=20)
plt.ylabel('Sepal width', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Sepal length vs. Sepal width',fontsize=20)
plt.legend(prop={'size':18})

plt.show()

plt.figure(1, figsize=(8,8))
plt.scatter(data[:50,2], data[:50,3], c="r", label="Iris-setosa")
plt.scatter(data[50:100,2], data[50:100,3], c="g", label="Iris-versicolor")
plt.scatter(data[100:,2], data[100:,3], c="b", label="Iris-virginica")
plt.xlabel('Sepal length', fontsize=15)
plt.ylabel('Sepal width', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Petal length vs. Petal width',fontsize=15)
plt.legend(prop={'size':20})

plt.show()

#see correlation matrix for all species for petal length and petal width all rows and all columns starting at 2 (petal-length)
print(dataset.iloc[:,2:].corr())

#we can also print correlation matrix for each species separately 0-50 50-100 100-150 for the three different types
#print(dataset.iloc[:50,:].corr())


#now we can visualize feature distribution
# we can see where there are multiple distributions from each of the species

fig=plt.figure(figsize=(8,8))

ax=fig.gca()

dataset.hist(ax=ax)

plt.show()

print(dataset.describe())








