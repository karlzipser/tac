## 79 ########################################################################

print(__file__)
assert 'project_' in __file__
from utilz2 import *
import sys,os
from projutils import *
from ..params.a import *
from .dataloader import *
from .stats import *
from ..net.code.net import *

thispath=pname(pname(__file__))
sys.path.insert(0,opj(thispath,'env'))
weights_path=opj(thispath,'net/weights')
figures_path=opj(thispath,'figures')
stats_path=opj(thispath,'stats')

mkdirp(figures_path)
mkdirp(weights_path)
mkdirp(stats_path)

weights_file=opj(weights_path,'latest.pth')
stats_file=opj(stats_path,'stats.txt')

device = torch.device(p.device if torch.cuda.is_available() else 'cpu')


class Loss_Recorder():
    def __init__(
        self,
        path,
        s=0.001,
    ):
        super(Loss_Recorder,self).__init__()
        packdict(self,locals())
        self.f=[]
        self.t=[]
        self.i=[]
        self.r=[]
        self.ctr=0
    def add(d):
        self.t.append(time.time())
        self.f.append(d)
        if ctr:
            s=self.s
            a=self.r[ctr-1]
            b=(1-s)*a+s*d
            self.r.append(b)
        else:
            self.r.append(d)
        self.ctr+=1
        self.i.append(ctr)
    def save():
        so(opj(self.path,'loss'),self.__dict__)
    def load():
        d=lo(opj(self.path,'loss'))
        for k in d:
            self.__dict__[k]=d[k]
    def plot():
        figure('loss')
        clf()
        plot(self.i,self.f,'c')
        plot(self.i,self.r,'b')
        plt.xlabel('iterations');
        plt.ylabel('loss')
        plt.title(self.path.replace(opjh(),''))
        plt.savefig(opj(self.path,'loss.pdf'))

loss_recorder=Loss_Recorder(thispath)

if p.run_path:
    print('****** Continuing from',p.run_path)
    net=get_net(
        device=device,
        run_path=p.run_path,
    )
else:
    net=get_net(device=device,net_class=Net)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=.001,betas=(0.5,0.999))

save_timer=Timer(p.save_time)
save_timer.trigger()
loss_timer=Timer(p.loss_time)
loss_ctr=0
loss_ctr_all=0
it_list=[]
running_loss_list=[]

print('*** Start Training . . .')

for epoch in range(p.num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs = net(inputs)
        loss = criterion(torch.flatten(outputs,1), labels)
        loss.backward()
        optimizer.step()
        loss_recorder.add(loss.item)
        loss_recorder.plot()
        if False:
            running_loss += loss.item()
            loss_ctr+=1
            loss_ctr_all+=1
            if loss_timer.rcheck():
                print(
                    f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / loss_ctr:.3f}')
                it_list.append(loss_ctr_all)
                running_loss_list.append(running_loss/loss_ctr)
                running_loss = 0.0
                loss_ctr=0
                if 'graphics':
                    figure(1)
                    clf()
                    plot(it_list,running_loss_list)
                    plt.xlabel('iterations');
                    plt.ylabel('avg loss')
                    plt.title(__file__.replace(opjh(),''))
                    plt.savefig(opj(figures_path,'loss.pdf'))
        if save_timer.rcheck():
            save_net(net,weights_file)

print('*** Finished Training')


save_net(net,weights_file)

net=get_net(device=device,net_class=Net,weights_file=weights_file)

"""
dataiter = iter(testloader)
images, labels = next(dataiter)
sh(torchvision.utils.make_grid(images),'grid')
plt.savefig(figure_file)
"""

stats=get_accuracy(net,testloader,classes,device)
print(stats)
t2f(stats_file,stats)

print('*** Done')

#EOF
## 79 ########################################################################
