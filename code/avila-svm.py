# Author: Julius Weißmann
# Github: weiJul

from preproData import get_data
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# data
trainData, trainLabels, testData, testLabels = get_data()

trainDataNP = np.asarray(trainData,  dtype=np.float64)
trainLabelsNP = np.asarray(trainLabels,  dtype=np.int)
testDataNP = np.asarray(testData,  dtype=np.float64)
testLabelsNP = np.asarray(testLabels,  dtype=np.int)

# svm
clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf'))
clf.fit(trainDataNP, trainLabelsNP)

out = clf.predict(testDataNP)

# print acc
counter = 0
for cnt, i in enumerate(testLabelsNP):
    if i==out[cnt]:
        counter+=1
acc = counter/len(testLabelsNP)
print("Accuracy: %f" % acc) # Accuracy: 0.703076
