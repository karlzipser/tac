## 79 ########################################################################
# branch master
print(__file__)
assert 'project_' in __file__
from utilz2 import *
import sys,os
from projutils import *
#from ..params.a_local import *
from .dataloader import *
#from .stats import *
from ..net.code.net import *

thispath=pname(pname(__file__))
sys.path.insert(0,opj(thispath,'env'))
weights_path=opj(thispath,'net/weights')
figures_path=opj(thispath,'figures')
stats_path=opj(thispath,'stats')

mkdirp(figures_path)
mkdirp(weights_path)
mkdirp(stats_path)

weights_latest=opj(weights_path,'latest.pth')
weights_best=  opj(weights_path,'best.pth')
#stats_file=opj(stats_path,'stats.txt')

device = torch.device(p.device if torch.cuda.is_available() else 'cpu')
kprint(p.__dict__)
best_loss=1e999

stats_recorders={}
stats_keys=['train loss','test loss',]
if p.run_path:
    print('****** Continuing from',p.run_path)
    net=get_net(
        device=device,
        run_path=p.run_path,
    )
    for k in stats_keys:
        stats_recorders[k]=Loss_Recorder(
            opjh(p.run_path,fname(thispath),'stats'),
            pct_to_show=p.percent_loss_to_show,
            s=p.loss_s,
            name=k,)
        stats_recorders[k].load()
        stats_recorders[k].path=stats_path
else:
    net=get_net(device=device,net_class=Net)
    for k in stats_keys:
        s=p.loss_s
        if 'test loss' in k:
            s*=p.test_sample_factor
        stats_recorders[k]=Loss_Recorder(
            stats_path,
            pct_to_show=p.percent_loss_to_show,
            s=s,
            name=k,
            )



stats_recorders['test accuracy']=Loss_Recorder(
    stats_path,
        plottime=0,
        savetime=0,
        sampletime=0,
        nsamples=1,
    pct_to_show=p.percent_loss_to_show,
    s=0.25,
    name='test accuracy',
    )

criterion=p.criterion
if p.opt==optim.Adam:
    optimizer = p.opt(net.parameters(),lr=p.lr)
else:
    optimizer = p.opt(net.parameters(),lr=p.lr,momentum=p.momentum)

save_timer=Timer(p.save_time)
test_timer=Timer(p.test_time)
max_timer=Timer(p.max_time)
show_timer=Timer(p.show_time);show_timer.trigger()
loss_ctr=0
loss_ctr_all=0
it_list=[]
running_loss_list=[]

print('*** Start Training . . .')

def show_sample_outputs(outputs,train_labels):
    outputs=outputs.detach().cpu().numpy()
    train_labels=train_labels.detach().cpu().numpy()
    print(outputs)
    print(train_labels)
    print(shape(outputs))
    print(shape(train_labels))
    for i in range(16):
        o=outputs[i,:,0,0]
        l=0*o
        l[train_labels[i]]=1
        clf()
        plot(o/o.max(),'r')
        plot(o,'k')
        plot(l,'b')
        cm()




from torch.utils.data import DataLoader, Dataset
class GenDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        fs0=sggo(opjD('data/gen0/*'))
        for cf in fs0:
            if not os.path.isdir(cf):
                continue
            for image in sggo(cf,'*.png'):
                self.images.append(image)
                self.labels.append(fname(cf))
                print(self.images[-1],self.labels[-1])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = rimread(self.images[index])
        #sh(z55(image),title=d2s(image.max(),image.min()),r=0)
        if self.transform:
            image = self.transform(image)
            #image-=0.5
            #image*=2.
        return image, self.labels[index]


gen_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]+geometric_transforms_list+color_transforms_list)
gen_traindata = GenDataset(root=opjD('data/gen0'), transform=gen_train_transform)
gen_trainloader = DataLoader(gen_traindata, batch_size=p.batch_size, shuffle=True)



classes2nums={}
for i in rlen(classes):
    classes2nums[classes[i]]=i




figure('train examples',figsize=(8,4))

for epoch in range(p.num_epochs):
    if max_timer.check():
        break
    kprint(
        files_to_dict(thispath),
        showtype=False,
        title=thispath,
        space_increment='....',)
    running_loss = 0.0

    train_dataiter = iter(trainloader)
    
    i=-1
    while i<50000:
        i+=1
        if randint(2)<1:
            try:
                #printr(1)
                train_inputs,train_labels=next(train_dataiter)
                #cb(train_inputs.min(),train_inputs.max())
            except:
                #printr(2)
                train_dataiter=iter(trainloader)
                train_inputs,train_labels=next(train_dataiter)
        else:
            try:
                #printr(3)
                train_inputs,train_labels=next(gen_train_dataiter)
                #cg(train_inputs.min(),train_inputs.max())
            except:
                #printr(4)
                gen_train_dataiter=iter(gen_trainloader)
                train_inputs,train_labels=next(gen_train_dataiter)
        if p.noise_level and rnd()<p.noise_p:
            train_inputs+=rnd()*p.noise_level*torch.randn(train_inputs.size())
        optimizer.zero_grad()
        train_inputs=train_inputs.to(device)
        if show_timer.rcheck():
            sh(torchvision.utils.make_grid(train_inputs),'train examples')
        outputs = net(train_inputs)
        targets=0*outputs.detach()
        for ii in range(targets.size()[0]):
            qq=train_labels[ii]
            if type(qq) is str:
                pass#qq=torch.tensor(classes2nums[qq])
            else: #cg(qq,r=1)
                targets[ii,qq,0,0]=1
        #show_sample_outputs(outputs,train_labels)
        loss = criterion(torch.flatten(outputs,1),torch.flatten(targets,1))
        loss.backward()
        optimizer.step()
        stats_recorders['train loss'].do(loss.item())
        if not i%p.test_sample_factor:
            #printr(i,'test')
            net.eval()
            try:
                test_inputs,test_labels = next(test_dataiter)
            except:
                test_dataiter = iter(testloader2)
                test_inputs,test_labels = next(test_dataiter)
            test_inputs=test_inputs.to(device)
            #if show_timer.rcheck():
            #    sh(torchvision.utils.make_grid(test_inputs),'test examples')
            test_labels=test_labels.to(device)
            test_outputs=net(test_inputs)
            #show_sample_outputs(test_outputs,test_labels)
            targets=0*test_outputs.detach()
            for ii in range(targets.size()[0]):
                targets[ii,test_labels[ii],0,0]=1
            test_loss=criterion(torch.flatten(test_outputs,1),torch.flatten(targets,1))
            if len(stats_recorders['train loss'].i):
                ec=external_ctr=stats_recorders['train loss'].i[-1]
            else:
                ec=0
            stats_recorders['test loss'].do(
                test_loss.item(),
                external_ctr=ec)
            net.train()
        if save_timer.rcheck():
            save_net(net,weights_latest)
            current_loss=stats_recorders['test loss'].current()
            
            if current_loss<=best_loss:
                best_loss=current_loss
                save_net(net,weights_best)
            else:
                print('*** current_loss=',current_loss,'best_loss=',best_loss)
            if not ope(weights_best):
                save_net(net,weights_best)
            
            fs=sggo(weights_path,'*.pth')
            tx=[d2s('epoch',epoch,'current_loss=',
                dp(current_loss,5),'best_loss=',dp(best_loss,5))]
            for f in fs:
                tx.append(
                    d2s(f,time_str(t=os.path.getmtime(f)),os.path.getsize(f)))
            t2f(opj(stats_path,'weights_info.txt'),'\n'.join(tx))
        if stats_recorders['train loss'].plottimer.rcheck():
            stats_recorders['train loss'].plot()
            stats_recorders['test loss'].plot(
                clear=False,rawcolor='y',smoothcolor='r',savefig=True)
            spause()

    stats,acc_mean=get_accuracy(net,testloader,classes,device)
    print(time_str('Pretty2'),'epoch=',epoch)
    print(stats)
    t2f(opj(stats_path,time_str()+'.txt'),
        d2s(time_str('Pretty2'),'epoch=',epoch,'\n\n')+stats)
    ec=external_ctr=stats_recorders['train loss'].i[-1]
    stats_recorders['test accuracy'].do(
        acc_mean,
        external_ctr=ec,)
    kprint(stats_recorders['test accuracy'].__dict__)
    stats_recorders['test accuracy'].plot(fig='test accuracy',savefig=True)
    spause()
print('*** Finished Training')

if False:
    save_net(net,weights_file)

    net=get_net(device=device,net_class=Net,weights_file=weights_file)

    stats=get_accuracy(net,testloader,classes,device)
    print(stats)
    t2f(stats_file,stats)

print('*** Done')

#EOF
## 79 ########################################################################
