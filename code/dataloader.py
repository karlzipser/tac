## 79 ########################################################################

import torch
import torchvision
import torchvision.transforms as transforms
from ..params.a_local import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=p.batch_size,
                                    shuffle=True, num_workers=p.num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testset2 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=p.batch_size,
                                    shuffle=True, num_workers=p.num_workers)
testloader2 = torch.utils.data.DataLoader(testset2, batch_size=p.batch_size,
                                    shuffle=True, num_workers=p.num_workers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#EOF
