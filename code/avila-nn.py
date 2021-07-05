# Author: Julius Weißmann
# Github: weiJul

import torch
import torch.nn as nn
import numpy as np
import copy
from net_bn import BnNN
from net_deep import DeepNN
from preproData import get_data
import matplotlib.pyplot as plt
import torch.optim as optim
import random
import time

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# start time
# start_time = time.time()

lrDrop = 1
recStep = 50
iterations = 8000
learn = 0.01
# model parameters
inputLayer = 10
outputLayer = 12

# transform data from .txt



trainData, trainLabels, testData, testLabels = get_data()

# train data to torch
trainDataNP = np.asarray(trainData,  dtype=np.float32)
trainLabelsNP = np.asarray(trainLabels,  dtype=np.int)
trainDataTorch = torch.from_numpy(trainDataNP).float()
trainLabelsTorch = torch.from_numpy(trainLabelsNP).type(torch.LongTensor)

# test data to torch
testDataNP = np.asarray(testData,  dtype=np.float32)
testLabelsNP = np.asarray(testLabels,  dtype=np.int)
testDataTorch = torch.from_numpy(testDataNP).float()

# select model
# model = BnNN(inputLayer, outputLayer)
model = DeepNN(inputLayer, outputLayer)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learn, momentum=0.3) # step_size=200 , learn = 0.1
# optimizer = torch.optim.SGD(model.parameters(), lr=learn, momentum=0.3) # step_size=1200, learn = 0.1
# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None) #step_size=20, learn = 0.1

# lr_scheduler = reducing lr by plateau
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=100, verbose=True)
# scheduler = reducing lr by epoch
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=700, gamma=0.95) # without Dropout step_size should be between 200 - 400

succ = 0
stat = []

best_acc = 0
acc_train_rec = []
acc_test_rec = []

# start training
for i in range(iterations):
    running_loss = 0.0
    # randomize data
    torch.manual_seed(i)
    trainDataTorch = trainDataTorch[torch.randperm(trainDataTorch.size()[0])]
    torch.manual_seed(i)
    trainLabelsTorch = trainLabelsTorch[torch.randperm(trainDataTorch.size()[0])]

    # train
    model.train()
    def closure():
        global out
        # running_loss = 0.0
        out = model(trainDataTorch)
        loss = criterion(out, trainLabelsTorch)
        # running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        # lr_scheduler.step(running_loss)
        scheduler.step()
        return loss
    optimizer.step(closure)
    predstr = torch.argmax(out, 1)


    trainLabnp = trainLabelsTorch.numpy()

    # test
    model.eval()

    out=model(testDataTorch)
    predsts = torch.argmax(out, 1)

    # print progress
    succtr = 0
    succts = 0
    if i%recStep==0:
        for cnt, j in enumerate(predstr):
            if j == trainLabnp[cnt]:
                succtr += 1
        acctr = succtr / len(predstr)
        acc_train_rec.append(acctr)
        print("Train")
        print("Step: ",i," acc: ", acctr)

        for cnt, j in enumerate(predsts):
            if j == testLabels[cnt]:
                succts+=1
        accts = succts/len(predsts)
        acc_test_rec.append(accts)
        print("Test")
        print("Step: ",i," acc: ", accts)
        print()


    # save best test acc
    if accts > best_acc:
        best_acc = accts
        best_model_wts = copy.deepcopy(model.state_dict())


torch.save(model.state_dict(), "avilaModel.pth")
print("best val acc: ", best_acc)

#print time
# print("--- %s seconds ---" % (time.time() - start_time))

# plot result
plt.title("Validation and Training Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")
plt.plot([(x+1)*recStep for x, i in enumerate(acc_train_rec)],acc_train_rec, color='blue',label="train acc")
plt.plot([(x+1)*recStep for x, i in enumerate(acc_test_rec)],acc_test_rec, color='green',label="test acc")
plt.legend()
plt.savefig("plots.png", bbox_inches='tight')
