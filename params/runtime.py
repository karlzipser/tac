from utilz2 import *
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
    #percent_loss_to_show=100,
    #loss_s=.01,
    #test_sample_factor=6,
    #criterion=nn.CrossEntropyLoss()
    #opt=optim.SGD,lr=0.001,momentum=0.9,
    criterion=nn.MSELoss(),
    opt=optim.Adam,lr=0.0001,momentum=None,
    #max_time=222222*hours,
    noise_level=1.,
    noise_p=1.,
    gen_data_path=opjD('data/gen0'),
)

for k in p.times.__dict__:
    p.timer.__dict__[k]=Timer(p.times.__dict__[k])
_proj_dict=dict(
    hiMac='project_tac/29Jun24_10h40m23s',
    jane='',
    jack='',
    gauss=''
)
assert host_name in _proj_dict
p.run_path=_proj_dict[host_name]

#EOF
#epoch_p.timer=p.timer(5*minutes)