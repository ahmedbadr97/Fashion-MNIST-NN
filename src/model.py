import torch
from torch import nn
import torch.nn.functional as F


class FashionMinstModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = F.rrelu(x)
        x=self.dropout(x)

        x = self.fc2(x)
        x = F.rrelu(x)
        x=self.dropout(x)

        x = self.fc3(x)
        x = F.log_softmax(x,dim=1)

        return x

    def predict(self,x):
        self.eval()
        with torch.no_grad():
            x=self.forward(x)
            x=torch.exp(x)
        self.train()
        return x

