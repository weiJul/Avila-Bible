# Author: Julius Weißmann
# Github: weiJul


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from preproData import get_data

# data
trainData, trainLabels, testData, testLabels = get_data()

trainDataNP = np.asarray(trainData,  dtype=np.float64)
trainLabelsNP = np.asarray(trainLabels,  dtype=np.int)
testDataNP = np.asarray(testData,  dtype=np.float64)
testLabelsNP = np.asarray(testLabels,  dtype=np.int)

# KNN
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(trainDataNP, trainLabels)

out = neigh.predict(testDataNP)

# print acc
counter = 0
for cnt, i in enumerate(testLabelsNP):
    if i==out[cnt]:
        counter+=1
acc = counter/len(testLabelsNP)
print("Accuracy: %f" % acc) # Accuracy: 0.757881

