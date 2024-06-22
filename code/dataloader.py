## 79 ########################################################################

import torch
import torchvision
import torchvision.transforms as transforms
from ..params.a_local import *
from projutils.data_augmentation import get_transforms

_fill=(0,0,0)

transforms_dict=dict(
    RandomPerspective=True,
    RandomPerspective_distortion_scale=0.3,
    RandomPerspective_p=0.3,
    RandomPerspective_fill=_fill,

    RandomRotation=True,
    RandomRotation_angle=12,
    RandomRotation_fill=_fill,

    RandomResizedCrop=True,
    RandomResizedCrop_scale=(0.85,1),
    RandomResizedCrop_ratio=(0.85,1.2),

    RandomHorizontalFlip=True,
    RandomHorizontalFlip_p=0.5,
        
    RandomVerticalFlip=False,
    RandomVerticalFlip_p=0.5,

    RandomZoomOut=True,
    RandomZoomOut_fill=_fill,
    RandomZoomOut_side_range=(1.0,1.5),

    ColorJitter=False,
    ColorJitter_brightness=(0,1),
    ColorJitter_contrast=(0,1),
    ColorJitter_saturation=(0,2),
    ColorJitter_hue=(-.03,.03),
)
transforms_dict2=dict(
    RandomPerspective=True,
    RandomPerspective_distortion_scale=0.5,
    RandomPerspective_p=0.5,
    RandomPerspective_fill=_fill,

    RandomRotation=True,
    RandomRotation_angle=16,
    RandomRotation_fill=_fill,

    RandomResizedCrop=True,
    RandomResizedCrop_scale=(0.75,1),
    RandomResizedCrop_ratio=(0.75,1.2),

    RandomHorizontalFlip=True,
    RandomHorizontalFlip_p=0.5,
        
    RandomVerticalFlip=False,
    RandomVerticalFlip_p=0.5,

    RandomZoomOut=True,
    RandomZoomOut_fill=_fill,
    RandomZoomOut_side_range=(1.0,1.5),

    ColorJitter=False,
    ColorJitter_brightness=(0,1),
    ColorJitter_contrast=(0,1),
    ColorJitter_saturation=(0,2),
    ColorJitter_hue=(-.03,.03),
)
geometric_transforms_list,color_transforms_list=get_transforms(
	d=transforms_dict2,
	image_size=(32,32))

train_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	]+geometric_transforms_list+color_transforms_list)
test_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=p.batch_size,
                                    shuffle=True, num_workers=p.num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=test_transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=p.batch_size,
                                    shuffle=True, num_workers=p.num_workers)

testset2 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=test_transform)
testloader2 = torch.utils.data.DataLoader(testset2, batch_size=p.batch_size,
                                    shuffle=True, num_workers=p.num_workers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#EOF
