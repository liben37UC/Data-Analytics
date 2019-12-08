from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

train_data,test_data,train_label,test_label=train_test_split(dataset.iloc[:,:3], dataset.iloc[:,4], test_size=0.2, random_state=42)

#initializing variables k having values from 1-9
neighbors=np.arange(1,9)

#two matrices for training and testing accuracy
train_accuracy=np.zeros(len(neighbors))

test_accuracy=np.zeros(len(neighbors))

for i,k in enumerate(neighbors):
	#i index, k is elem
	#initialize KNN for k neighbors
	knn=KNeighborsClassifier(n_neighbors=k)
	#fit the model with the train data
	knn.fit(train_data,train_label)
	#compute train accuracy
	train_accuracy[i]=knn.score(train_data,train_label)
	#compute accuracy on test
	test_accuracy[i]=knn.score(test_data,test_label)

plt.figure(figsize=(10,6))
plt.title('KNN accuracy with varying number of neighbors', fontsize=20)
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend(prop={'size':20})
plt.xlabel('Number of neighbors',fontsize=20)
plt.ylim(0,1)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()









