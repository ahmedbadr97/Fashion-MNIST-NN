import torch
from torch import nn
import torch.nn.functional as F


class FashionMinstModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        self.dropout(x)

        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)

        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
            x = torch.exp(x)
        self.train()
        _, top_class = x.topk(1, dim=1)
        top_class = top_class.view(-1)
        return top_class

    def load_weights(self, path):
        state_dict=torch.load(path)
        self.load_state_dict(state_dict)

