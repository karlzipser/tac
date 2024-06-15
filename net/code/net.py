import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
net_path=__file__

class Net(nn.Module):
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

if False:
    from project_tac.a14Jun24_23h26m41s_with_net.tac.net.code.net import *
    w=most_recent_file_in_folder(opj(pname(pname(net_path)),'weights'))
    from UTILS_.vis import *
    #w='/Users/karlzipser/project_tac/14Jun24_23h35m34s_with_net/tac/net/weights/14Jun24_23h35m37s.pth'
    net=get_net(Net,weights_file=w)

if False:
    # load net given only the project path minimal information
    import importlib.util
    from UTILS_.vis import * # import this from local env
    file_path = '/Users/karlzipser/project_tac/a14Jun24_23h26m41s_with_net'
    # add the rest of the paths for net.py and weights '/net/code/net.py'
    module_name = 'module'
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    w=most_recent_file_in_folder(opj(pname(pname(module.net_path)),'weights'))
    net=get_net(module.Net,weights_file=w)

#EOF
