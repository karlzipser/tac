from utilz2 import *
import projutils

_t=30
p=k2c(
    ti='p',
    batch_size=16,
    num_workers=8,
    num_epochs=100000,
    device='cuda:0',
    times=k2c(
        save=_t*10,
        show=_t,
        epoch=95*minutes,
        max=999*hours,
    ),
    timer=k2c(ti='timer'),
    #criterion=nn.CrossEntropyLoss()
    #opt=optim.SGD,lr=0.001,momentum=0.9,
    criterion=nn.MSELoss(),
    opt=optim.Adam,lr=0.0001,momentum=None,
    gen_data_path=opjD('data/gen0'),
    task_list=5*['train']+1*['test']+1*['train_no_noise'],
    data_recorders=dict(
        train=projutils.net_data_recorder.Data_Recorder(
            dataloader='trainloader',
            name='train',
            noise_level=1.,
            noise_p=1.,
            ),
        train_no_noise=projutils.net_data_recorder.Data_Recorder(
            dataloader='trainloader',
            name='train_no_noise',
            noise_level=0.,
            noise_p=0.,
            ),
        test=projutils.net_data_recorder.Data_Recorder(
            dataloader='testloader',
            name='test',
            noise_level=0.,
            noise_p=0.,
            ),
        ),
)

for k in p.times.__dict__:
    p.timer.__dict__[k]=Timer(p.times.__dict__[k])
_proj_dict=dict(
    hiMac='project_tac/29Jun24_15h19m10s',
    jane='project_tac/29Jun24_12h22m37s',
    jack='',
    gauss='project_tac/29Jun24_13h50m54s',
)
assert host_name in _proj_dict
p.run_path=_proj_dict[host_name]

#EOF
#epoch_p.timer=p.timer(5*minutes)