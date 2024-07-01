### ulimit -n 8192
##############################################################################
##
from utilz2 import *
import projutils

_t=1*minute
_n=100

p=k2c(
    ti='p',
    batch_size=1,
    num_workers=8,
    num_epochs=100000,
    device='cuda:0',
    image_width=128,
    times=k2c(
        save=_t,#*10,
        show=_t,
        epoch=2*minutes,
        max=999*hours,
    ),
    timer=k2c(ti='timer'),
    max_num_samples=50,
    ##criterion=nn.CrossEntropyLoss()
    ##opt=optim.SGD,lr=0.001,momentum=0.9,
    criterion=nn.MSELoss(),
    opt=optim.Adam,lr=0.0001,momentum=None,
    gen_data_path=opjD('data/rf_gen128_0'),
    task_list=5*['train']+1*['test']+5*['gen_trainloader'],
    ##task_list=5*['train']+1*['test'],
    run_path=select_from_list(sggo(opjh('project_tac/*'))),
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
            noise_p=.75,
            targets_to_zero=True,
            n=_n,
            )

for k in p.times.__dict__:
    p.timer.__dict__[k]=Timer(p.times.__dict__[k])

##
##############################################################################
##
#EOF
