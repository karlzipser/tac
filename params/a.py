from utilz2 import *

p=k2c(
	batch_size=16,
	num_workers=4,
	num_epochs=100000,
	device='cuda:0',
	save_time=60,
	loss_time=10,
	test_time=60,
	run_path='',#'project_tac/17Jun24_22h35m42s-net_net', #'project_tac/17Jun24_21h48m18s-net_net',
)

#EOF
