# Author: Julius Weißmann
# Github: weiJul

def get_data():
    train_file = open("avila/avila-tr.txt", "r")
    test_file = open("avila/avila-ts.txt", "r")
    files = [train_file, test_file]

    targetsChar = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "W", "X", "Y"]
    trainData = []
    trainLabels =[]
    testData = []
    testLabels =[]
    countera = 0
    for f in files:
        for count, line in enumerate(f):
            lineArr = line.split(",")
            label = lineArr.pop()
            label = targetsChar.index(label[0])
            if f == files[0]:
                trainData.append(lineArr)
                trainLabels.append(label)
            else:
                testData.append(lineArr)
                testLabels.append(label)




    return trainData, trainLabels, testData, testLabels
