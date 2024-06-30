## 79 ########################################################################
from utilz2 import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
net_path=__file__




def shape_from_tensor(x):
    return shape( x.cpu().detach().numpy() )


def cuda_to_rgb_image(cu):
    if len(cu.size()) == 3:
        return z55(cu.detach().cpu().numpy().transpose(1,2,0))
    elif len(cu.size()) == 4:
        return z55(cu.detach().cpu().numpy()[0,:].transpose(1,2,0))
    else:
        assert False


def f___weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Describe_Layer(nn.Module):
    def __init__(
        _,
        name,
        show,
        endl='',
    ):
        super( Describe_Layer, _).__init__()
        _.name = name
        _.show = show
        _.first_pass = True
        _.endl = endl
    def forward(_, x):
        if _.first_pass and _.show:
            s = shape_from_tensor(x)
            print(_.name+':',s,dp(s[1]*s[2]*s[3]/1000.,1),'\bk',_.endl)
            _.first_pass = False
        return x
dl=Describe_Layer

#bceloss = nn.BCELoss()

#mseloss = nn.MSELoss()



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

class Net32(nn.Module):
    def __init__(self):
        n=3
        super().__init__()
        self.a = nn.Conv2d(3,n*16,kernel_size=5,stride=2,padding=1)
        self.b = nn.Conv2d(n*16,n*64,kernel_size=5,stride=2,padding=1)
        self.c = nn.Conv2d(n*64,n*128,kernel_size=5,stride=2,padding=1)
        self.d = nn.Conv2d(n*128,n*256,kernel_size=5,stride=2,padding=1)
        self.e = nn.Conv2d(n*256, 10,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        #print(1,x.size())
        x = F.relu(self.a(x))
        #print(2,x.size())
        x = F.relu(self.b(x))
        #print(3,x.size())
        x = F.relu(self.c(x))
        #print(4,x.size())
        x = self.d(x)
        #print(5,x.size())
        x = self.e(x)
        #print(6,x.size())
        #input('here')
        return x



class Net128(nn.Module):
    def __init__(
        _,
        nin=3,
        ndf=16,
        nout=10,
        ):
        super(Net128, _).__init__()

        _.main = nn.Sequential(                   
            nn.Identity(),                                             dl('\nDiscriminator128 input',True),
                        
            nn.Conv2d(nin, ndf//4, 4, 2, 1),                           dl('\tConv2d output',True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf//4, ndf, 4, 2, 1, bias=False),               dl('\tConv2d output',True),
            nn.BatchNorm2d( ndf ),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 4, 4, 2, 1, bias=False),              dl('\tConv2d output',True),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 16, 4, 2, 1, bias=False),         dl('\tConv2d output',True),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),        dl('\tConv2d output',True),
            nn.BatchNorm2d(ndf * 32),\
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 32, nout, 4, 1, 0),                        dl('\tConv2d output',True),
            #nn.Upsample((1,1),mode='nearest'),                        dl('\tUpsample output',True,'\n'),                
            #nn.Sigmoid(),                                              dl('\tSigmoid',True,'\n'),
        )
    def forward(_, x):
        #assert x.size()[-1]==128
        return _.main(x)


Net=Net128



#EOF
