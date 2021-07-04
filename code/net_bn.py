# Author: Julius Weißmann
# Github: weiJul

import torch.nn as nn
import torch.nn.functional as F

class BnNN(nn.Module):
    def __init__(self, inputLayer, outputLayer):
        super().__init__()
        self.linear1 = nn.Linear(inputLayer,500)
        self.linear2 = nn.Linear(500,12)
        self.linear3 = nn.Linear(12,outputLayer)
        self.bn2 = nn.BatchNorm1d(12)
        self.bn1 = nn.BatchNorm1d(500)

    def forward(self,x):
        x = F.relu6(self.bn1(self.linear1(x)))
        x = F.relu6(self.bn2(self.linear2(x)))
        x = self.linear3(x)

        return F.log_softmax(x, dim=1)
