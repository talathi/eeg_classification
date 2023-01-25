### Scripts to run hyper-parameter search for drl mask training

## Expt 1: Investigate random search for: num-frames stacking and ski-frames

import numpy as np
import sys,os
import time

def gpu_mem(index):
	str_cmd='nvidia-smi -i %d -q -d MEMORY |grep Used >tmp.txt'%index
	os.system(str_cmd)
	mem=open('tmp.txt').readlines()[0].split(': ')[1].split(' ')[0]
	return int(mem)

def gpu_under_use(index):
	mem=gpu_mem(index)
	if mem>1000:
		bul=True
	else:
		bul=False
	return bul

img_size_x=[32,64,128,256]
img_size_y=[40,80,160,320]
#skip_frames=[np.random.randint(1,num_frames[i]+1) for i in range(len(num_frames))]

count=0
while count<10:
	for k in range(4):
		if not gpu_under_use(3-k):
			print count,k
			str_cmd='THEANO_FLAGS=\'device=gpu%d,lib.cnmem=0.8\' python eeg_classification --train --learning-rate 0.001 --cool --epochs 300 --window-size 80\
			 --save-dir /home/talathi1/Work/Python/Models/keras_models/rnneeg_models --memo rnn_eeg_model_case0_w80_300epochs_%d\
			>> /home/talathi1/Work/Python/Models/keras_models/conveeg_models/logfiles/rnn_eeg_model_case0_w80_300epochs_%d.log 2>&1&'%(k,count,count)
			print str_cmd
			os.system(str_cmd)
			time.sleep(10)
			count+=1

