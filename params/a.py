from utilz2 import *

p=k2c(
	batch_size=16,
	num_workers=4,
	num_epochs=100000,
	device='cuda:0',
	save_time=60,
	loss_time=10,
	test_time=60,
	lr=0.0001,
	percent_loss_to_show=100,
	loss_s=.01,
	run_path='',#'project_tac/18Jun24_22h52m56s',#'project_tac/18Jun24_14h09m36s_long_train' #'project_tac/18Jun24_13h59m32s',
)

#EOF
