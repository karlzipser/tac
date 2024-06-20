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
stats_file=opj(stats_path,'stats.txt')

device = torch.device(p.device if torch.cuda.is_available() else 'cpu')
kprint(p.__dict__)
best_loss=1e999

if p.run_path:
    print('****** Continuing from',p.run_path)
    net=get_net(
        device=device,
        run_path=p.run_path,
    )
    loss_recorder_train=Loss_Recorder(
        opjh(p.run_path,fname(thispath),'stats'),
        pct_to_show=p.percent_loss_to_show,
        s=p.loss_s,
        name='train loss',)
    loss_recorder_train.load()
    loss_recorder_train.path=stats_path
    loss_recorder_test=Loss_Recorder(
        opjh(p.run_path,fname(thispath),'stats'),
        pct_to_show=p.percent_loss_to_show,
        s=p.loss_s,
        name='test loss',)
    loss_recorder_test.load()
    loss_recorder_test.path=stats_path
else:
    net=get_net(device=device,net_class=Net)
    loss_recorder_train=Loss_Recorder(
        stats_path,
        pct_to_show=p.percent_loss_to_show,
        s=p.loss_s,
        name='train loss',
        )
    loss_recorder_test=Loss_Recorder(
        stats_path,
        pct_to_show=p.percent_loss_to_show,
        s=p.loss_s*p.test_sample_factor,
        name='test loss',
        )

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=p.lr)

save_timer=Timer(p.save_time)
test_timer=Timer(p.test_time)
loss_ctr=0
loss_ctr_all=0
it_list=[]
running_loss_list=[]

print('*** Start Training . . .')

for epoch in range(p.num_epochs):
    kprint(
        files_to_dict(thispath),
        showtype=False,
        title=thispath,
        space_increment='....',)
    running_loss = 0.0
    dataiter = iter(testloader)
    for i, data in enumerate(trainloader, 0):
        printr(i,'train')
        inputs, labels = data
        optimizer.zero_grad()
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs = net(inputs)
        loss = criterion(torch.flatten(outputs,1), labels)
        loss.backward()
        optimizer.step()
        loss_recorder_train.do(loss.item())
        if not i%p.test_sample_factor:
            printr(i,'test')
            net.eval()
            test_inputs,test_labels = next(dataiter)
            test_outputs=net(test_inputs)
            test_loss=criterion(torch.flatten(test_outputs,1),test_labels)
            if len(loss_recorder_train.i):
                ec=external_ctr=loss_recorder_train.i[-1]
            else:
                ec=0
            loss_recorder_test.do(
                test_loss.item(),
                external_ctr=ec)
            net.train()
        if save_timer.rcheck():
            save_net(net,weights_latest)
            current_loss=loss_recorder_train.current()
            
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
        if loss_recorder_train.plottimer.rcheck():
            loss_recorder_train.plot()
            loss_recorder_test.plot(
                clear=False,rawcolor='g',smoothcolor='r',savefig=True)
            spause()
    if test_timer.rcheck():
        stats=get_accuracy(net,testloader,classes,device)
        print(time_str('Pretty2'),'epoch=',epoch)
        print(stats)
        t2f(stats_file,stats)

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
