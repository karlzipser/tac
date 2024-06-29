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

def show_sample_outputs(outputs,labels):
    outputs=outputs.detach().cpu().numpy()
    labels=labels.detach().cpu().numpy()
    print(outputs)
    print(labels)
    print(shape(outputs))
    print(shape(labels))
    for i in range(16):
        o=outputs[i,:,0,0]
        l=0*o
        l[labels[i]]=1
        clf()
        plot(o/o.max(),'r')
        plot(o,'k')
        plot(l,'b')
        cm()

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
    dataiter = iter(testloader2)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        #print(p.noise_level and rnd()<p.noise_p)
        if p.noise_level and rnd()<p.noise_p:
            inputs+=rnd()*p.noise_level*torch.randn(inputs.size())
        optimizer.zero_grad()
        inputs=inputs.to(device)
        if show_timer.rcheck():
            sh(torchvision.utils.make_grid(inputs),'train examples')
        outputs = net(inputs)
        targets=0*outputs.detach()
        for ii in range(targets.size()[0]):
            targets[ii,labels[ii],0,0]=1
        #show_sample_outputs(outputs,labels)
        loss = criterion(torch.flatten(outputs,1),torch.flatten(targets,1))# labels)
        loss.backward()
        optimizer.step()
        stats_recorders['train loss'].do(loss.item())
        if not i%p.test_sample_factor:
            #printr(i,'test')
            net.eval()
            test_inputs,test_labels = next(dataiter)
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
