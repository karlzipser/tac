## 79 ########################################################################

import torch
import torchvision
import torchvision.transforms as transforms
from ..params.runtime import *
from projutils.data_augmentation import get_transforms
from torch.utils.data import DataLoader, Dataset

classes = dict(
    plane=0,
    car=1,
    bird=2,
    cat=3,
    deer=4,
    dog=5,
    frog=6,
    horse=7,
    ship=8,
    truck=9,
    )
    
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
IMAGE_WIDTH=128
geometric_transforms_list,color_transforms_list=get_transforms(
	d=transforms_dict,
	image_size=(IMAGE_WIDTH,IMAGE_WIDTH))

train_transform = transforms.Compose([
	transforms.ToTensor(),
    transforms.Resize((IMAGE_WIDTH,IMAGE_WIDTH)),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	]+geometric_transforms_list)#+color_transforms_list)
test_transform = transforms.Compose([
    
	transforms.ToTensor(),
    transforms.Resize((IMAGE_WIDTH,IMAGE_WIDTH)),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform,
    )
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=p.batch_size,
    shuffle=True,
    num_workers=p.num_workers,
    )
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
    )
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=p.batch_size,
    shuffle=True,
    num_workers=p.num_workers,
    )

trainset_test_transform = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=test_transform,
    )
trainloader_test_transform = torch.utils.data.DataLoader(
    trainset_test_transform,
    batch_size=p.batch_size,
    shuffle=True,
    num_workers=p.num_workers,
    )


class GenDataset(Dataset):
    def __init__(self, root, transform=None):
        cy('GenDataset __init__()')
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        fs0=sggo(root,'*')
        for cf in fs0:
            if not os.path.isdir(cf):
                continue
            for image in sggo(cf,'*.png'):
                self.images.append(image)
                self.labels.append(fname(cf))
                #print(self.images[-1],self.labels[-1])
        print('len(self.images)=',
            len(self.images),'len(self.labels)=',len(self.labels))
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = rimread(self.images[index])
        if self.transform:
            image = self.transform(image)
        return image,classes[self.labels[index]]

#del inputs,labels
gen_traindata = GenDataset(
    root=p.gen_data_path,#pjD('data/gen0'),
    transform=train_transform)
gen_trainloader = DataLoader(
    gen_traindata, batch_size=16, shuffle=True)


loader_dic=dict(
    trainloader=trainloader,
    testloader=testloader,
    gen_trainloader=gen_trainloader,
    )






        
#EOF
