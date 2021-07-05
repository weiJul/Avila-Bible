import torch.nn as nn
import torch.nn.functional as F

class DeepNN(nn.Module):
    def __init__(self, inputLayer, outputLayer):
        super().__init__()
        self.linear1 = nn.Linear(inputLayer,100)
        self.linear2 = nn.Linear(100,60)
        self.linear3 = nn.Linear(60,40)
        self.linear4 = nn.Linear(40,12)
        self.linear5 = nn.Linear(12,outputLayer)
        self.dr = nn.Dropout(p=0.1)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(60)
        self.bn3 = nn.BatchNorm1d(40)
        self.bn4 = nn.BatchNorm1d(12)

    def forward(self,x):
        x = F.relu6(self.bn1(self.linear1(x)))
        x = self.dr(x)
        x = F.relu6(self.bn2(self.linear2(x)))
        x = self.dr(x)
        x = F.relu6(self.bn3(self.linear3(x)))
        x = self.dr(x)
        x = F.relu6(self.bn4(self.linear4(x)))
        x = self.dr(x)
        x = self.linear5(x)

        return F.log_softmax(x, dim=1)