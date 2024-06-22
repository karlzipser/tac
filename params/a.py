## 79 ########################################################################
assert False
from utilz2 import *
p=k2c(
	batch_size=16,
	num_workers=8,
	num_epochs=100000,
	device='cuda:0',
	save_time=60,
	loss_time=10,
	test_time=60,
	percent_loss_to_show=100,
	loss_s=.01,
	test_sample_factor=6,
	run_path='',
	criterion=nn.MSELoss(),
	#criterion=nn.CrossEntropyLoss()
	#opt=optim.SGD,lr=0.001,momentum=0.9,
	opt=optim.Adam,lr=0.0001,momentum=None,
	max_time=1*hours,
	show_time=20,
	noise_level=1.,
	noise_p=.75
)

#EOF


#EOF
