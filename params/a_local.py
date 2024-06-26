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
	criterion=nn.MSELoss(),
	#criterion=nn.CrossEntropyLoss()
	#opt=optim.SGD,lr=0.001,momentum=0.9,
	opt=optim.Adam,lr=0.0001,momentum=None,
	max_time=222222*hours,
	show_time=5,
	noise_level=0.,#1.,
	noise_p=.75
)
if host_name=='hiMac':
	p.run_path='project_tac/25Jun24_08h03m23s'
elif host_name=='jane':
	p.run_path='project_tac/25Jun24_08h03m23s'
	#'project_tac/24Jun24_23h48m30s'
	#'project_tac/24Jun24_23h04m02s'#'project_tac/21Jun24_22h28m24s'

#EOF
