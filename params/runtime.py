### ulimit -n 8192
##############################################################################
##
from utilz2 import *
import projutils

_t=1*minute
_n=100
p=k2c(
    ti='p',
    batch_size=16,
    num_workers=8,
    num_epochs=100000,
    device='cuda:0',
    times=k2c(
        save=_t,#*10,
        show=_t,
        epoch=10*minutes,
        max=999*hours,
    ),
    timer=k2c(ti='timer'),
    #criterion=nn.CrossEntropyLoss()
    #opt=optim.SGD,lr=0.001,momentum=0.9,
    criterion=nn.MSELoss(),
    opt=optim.Adam,lr=0.0001,momentum=None,
    gen_data_path=opjD('data/rf_gen128_0'),
    task_list=5*['train']+1*['test']+5*['gen_trainloader'],
    #task_list=5*['train']+1*['test'],

    data_recorders=dict(
        train=projutils.net_data_recorder.Data_Recorder(
            dataloader='trainloader',
            name='train',
            noise_level=1.,
            noise_p=1.,
            n=_n,
            ),

        test=projutils.net_data_recorder.Data_Recorder(
            dataloader='testloader',
            name='test',
            noise_level=0.,
            noise_p=0.,
            n=_n,
            ),
        ),
)


    
if 'gen_trainloader' in p.task_list:
    p.data_recorders[
        'gen_trainloader']=projutils.net_data_recorder.Data_Recorder(
            dataloader='gen_trainloader',
            name='gen_trainloader',
            noise_level=1.,
            noise_p=1.,
            targets_to_zero=True,
            n=_n,
            )




for k in p.times.__dict__:
    p.timer.__dict__[k]=Timer(p.times.__dict__[k])
_proj_dict=dict(
    hiMac='project_tac/30Jun24_11h15m55s',
    jane='',
    jake='',
    gauss='project_tac/30Jun24_12h02m38s',
)
assert host_name in _proj_dict
p.run_path=_proj_dict[host_name]

##
##############################################################################
##
#EOF
