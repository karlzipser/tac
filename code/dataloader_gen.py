from ..params.runtime import *

from torch.utils.data import DataLoader, Dataset

class GenDataset(Dataset):
    def __init__(self, root, transform=None):
        cy('GenDataset __init__()')
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        fs0=sggo(p.gen_data_path,'*')
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
        return dict(
                image=image,
                labels=self.labels[index],
                file=self.images[index],
                index=index,
            )


