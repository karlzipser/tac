from utilz2 import *

p=k2c(
	batch_size=16,
	num_workers=4,
	num_epochs=1000,
	device='cuda:0',
	save_time=60,
	loss_time=60,
)

#EOF
