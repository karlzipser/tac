import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
net_path=__file__

class Net_original(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3,8,kernel_size=3,stride=2,padding=1)
        self.b = nn.Conv2d(8,16,kernel_size=3,stride=2,padding=1)
        self.c = nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1)
        self.d = nn.Conv2d(32, 10,kernel_size=3,stride=2,padding=1)

    def forward(self, x):
        x = F.relu(self.a(x))
        x = F.relu(self.b(x))
        x = F.relu(self.c(x))
        x = self.d(x)

        return x
#EOF
