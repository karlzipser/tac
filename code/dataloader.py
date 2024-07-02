## 79                                                                       ##
##############################################################################
## 
#

##                                                                          ##
##############################################################################
##   
from utilz2 import *                                                        
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from ..params.runtime import *
#p=k2c(batch_size=3)
experiments=dict(
    phenomena=dict(
            AJ=dict(
                yes=['AJ'],
                no=['dynamic','PB'],
                ),
            PB=dict(
                yes=['PB'],
                no=['dynamic','AJ'],
                ),
            other=dict(
                yes=[],
                no=['dynamic','AJ','PB'],
            ),
        ),
    conditions=dict(
            DUG=dict(
                yes=['Dense','Urban','Ground'],
                no=['dynamic'],
                ),
            DUF=dict(
                yes=['Dense','Urban','Flight'],
                no=['dynamic'],
                ),
            OQ=dict(
                yes=['Open','Quiet'],
                no=['dynamic'],
                ),
            OUG=dict(
                yes=['Open','Urban','Ground'],
                no=['dynamic'],
                ),
            OUF=dict(
                yes=['Open','Urban','Flight'],
                no=['dynamic'],
                ),
        ),
    )

def file_matches(f,yesno):
    for q in yesno['yes']:
        if q not in f:
            return False
    for q in yesno['no']:
        if q in f:
            return False
    return True 

def filter_files_for_experiment(fs,experiment):
    d={}
    for f in fs:
        for k in experiment:
            yesno=experiment[k]
            if file_matches(f,yesno):
                if k not in d:
                    d[k]=[]
                d[k].append(f)
                continue
    return d

def d_to_lists(d):
    npy_file_list=[]
    label_list=[]
    ctr=0
    for k in d:
        for f in d[k]:
            npy_file_list.append(f)
            label_list.append(ctr)
        ctr+=1
    return npy_file_list,label_list



class RFDataset(Dataset):
    def __init__(self, npy_file_list, label_list, transform=transforms.ToTensor(),to3=True):
        print('\n*** RFDataset __init__()')
        assert len(npy_file_list)==len(label_list)
        self.transform=transform
        self.npy_file_list=npy_file_list
        self.label_list=label_list
        self.to3=to3
        print('RFDataset',len(self.npy_file_list),self.label_list)
        if False:
            fs0=sggo(root,'*')
            for cf in fs0:
                if not os.path.isdir(cf):
                    continue
                for image in sggo(cf,'*.npy'):
                    self.images.append(image)
                    self.labels.append(fname(cf))
            print('\tlen(self.images)=',
                len(self.images),'len(self.labels)=',len(self.labels))

    def __len__(self):
        return len(self.npy_file_list)

    def __getitem__(self, index):
        a,b=-2,8
        image = np.load(self.npy_file_list[index])
        m=np.concatenate((image[:16,:],image[-16:,:]))
        image-=m.mean()
        image/=m.std()
        image[image<a]=a
        image[image>b]=b
        image-=a
        image/=(b-a)
        image-=0.5
        image*=2.
        image[image<-1]=-1
        image[image>1]=1
        offset=randint(0,512-128)
        image=image[:,offset:offset+128]
        if self.to3:
            g=zeros((128,128,3))
            for i in range(3):
                g[:,:,i]=image
            image=g
        image=self.transform(image).float()
        return image,self.label_list[index]#,self.npy_file_list[index]



fs=find_files(opjD('data/RF/spectrograms4'),['*.npy'])
d=filter_files_for_experiment(fs,experiments['phenomena'])#'conditions'])#
a,b=d_to_lists(d)
c=list(rlen(a))
np.random.shuffle(c)
npy_file_list,label_list=[],[]
for i in c:
    npy_file_list.append(a[i])
    label_list.append(b[i])
n=int(0.8*len(npy_file_list))


trainloader=DataLoader(
    RFDataset(
        npy_file_list[:n], label_list[:n],
        ),
    batch_size=p.batch_size,
    shuffle=True)

testloader=DataLoader(
    RFDataset(
        npy_file_list[n:], label_list[n:],
        ),
    batch_size=p.batch_size,
    shuffle=True)


loader_dic=dict(
    trainloader=trainloader,
    testloader=testloader,
    )




classes=dict(
    DUG=0,
    DUF=1,
    OQ=2,
    OUG=3,
    OUF=4)
classes = dict(
    AJ=0,
    PB=1,
    other=2,
    )


if __name__ == '__main__':
    it=iter(testloader)
    ctr=0
    while True:
        try:
            inputs,labels=next(it)
            print(inputs.size(),labels)
            q=1*inputs
            q[0,0,0,0]=-1;q[0,0,0,1]=1
            sh(q[0,:],1,r=0)
            ctr+=1
        except StopIteration:
            print("Reached the end of the iterator.")
            break
    cg(ctr)
##                                                                          ##
##############################################################################
##                                                                       ##
it=iter(trainloader);inputs,labels=next(it);print(labels)


