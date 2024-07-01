### ulimit -n 8192
##############################################################################
##
from utilz2 import *
import projutils

def get_second_most_recent_file(folder_path):
    files = sggo(folder_path,'*')
    if len(files) < 2:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[1]



##                                                                          ##
##############################################################################
##                                                                          ##
def get_files_sorted_by_mtime(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort(key=lambda x: os.path.getmtime(x))
    return files
def select_files_to_keep(files, m):
    if m < 2:
        raise ValueError("The number of files to keep must be at least 2.")
    keep_files = [files[0], files[-1]]
    if m > 2:
        step = (len(files) - 1) / (m - 1)
        for i in range(1, m - 1):
            keep_files.append(files[int(round(i * step))])
    return sorted(keep_files, key=lambda x: os.path.getmtime(x))
def cleanup_folder(folder_path, keep_files):
    for f in os.listdir(folder_path):
        file_path = os.path.join(folder_path, f)
        if os.path.isfile(file_path) and file_path not in keep_files:
            os.remove(file_path)
def reduce_file_numbers(folder_path, num_to_keep):
    m=num_to_keep
    files = get_files_sorted_by_mtime(folder_path)
    keep_files = select_files_to_keep(files, m)
    cleanup_folder(folder_path, keep_files)
##                                                                          ##
##############################################################################
##                                                                          ##





_t=1*minute
_n=100
p=k2c(
    ti='p',
    batch_size=1,
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
    max_num_samples=50,
    #criterion=nn.CrossEntropyLoss()
    #opt=optim.SGD,lr=0.001,momentum=0.9,
    criterion=nn.MSELoss(),
    opt=optim.Adam,lr=0.0001,momentum=None,
    gen_data_path=opjD('data/rf_gen128_0'),
    task_list=5*['train']+1*['test']+5*['gen_trainloader'],
    #task_list=5*['train']+1*['test'],
    run_path=select_from_list(sggo(opjh('project_tac/*'))),#+['NONE']),#get_second_most_recent_file(opjh('project_tac')),
    #'project_tac/30Jun24_18h33m11s',#most_recent_file_in_folder(opjh('project_tac')),
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
"""
_proj_dict=dict(
    hiMac='project_tac/30Jun24_11h15m55s',
    jane='',
    jake='',
    gauss='project_tac/30Jun24_17h49m26s',
)
assert host_name in _proj_dict
p.run_path=_proj_dict[host_name]
"""
##
##############################################################################
##
#EOF
