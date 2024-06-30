test_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
    )
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=16,
    shuffle=True,
    num_workers=1,
    )



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

dataiter=iter(testloader)
inputs,labels=next(dataiter)
print(inputs.size(),inputs.min(),inputs.max())
print(labels)
"""
torch.Size([16, 3, 32, 32])
tensor([9, 8, 5, 9, 4, 0, 7, 7, 6, 8, 0, 3, 8, 9, 3, 7])
"""
#, a
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

gen_traindata = GenDataset(
    root=opjD('data/gen0'), transform=test_transform)
gen_trainloader = DataLoader(
    gen_traindata, batch_size=16, shuffle=True)
gen_train_dataiter=iter(gen_trainloader)
inputs2,labels2=next(gen_train_dataiter)
print(inputs2.size(),inputs2.min(),inputs2.max())
print(labels2)
#,b





#,a
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
def q(
    train=False,
    transform=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ]),
    batch_size=16,
    shuffle=True,
    num_workers=1,
    root='./data',
    ):
    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transform
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        )
    return loader
loader=q()
it=iter(loader)
for i in range(1000):
    a,b=next(it)
    sh(a)
    print(a.size(),a.min(),a.max())
    cm(b,r=1)
#,b