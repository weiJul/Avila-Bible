# Author: Julius Weißmann
# Github: weiJul

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from preproData import get_data

# get data
trainData, trainLabels, testData, testLabels = get_data()

# dt
clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(trainData, trainLabels)

# predict the response for test dataset
testDataPred = clf.predict(testData)

# model accuracy
print("Accuracy: %f" % metrics.accuracy_score(testLabels, testDataPred)) # Accuracy: 0.970202

