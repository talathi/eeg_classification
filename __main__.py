### 
# EEG Seizure classification using LSTM/GRU models with stateful boolean
#from __future__ import absolute_import
#from __future__ import print_function
import theano
import numpy as np
import pickle as PK
import sys
import time
import os,sys
import optparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import pandas as pd
import pickle
HOME=os.environ['HOME']
import matplotlib
matplotlib.use('TkAgg')
import pylab as py
py.ion()

### Add File Path
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)


if __name__=='__main__':

	parser=optparse.OptionParser()
	parser.add_option("--train", action="store_true",dest="train_bool",default=False,help="Invoke training")
	parser.add_option("--test", action="store_true",dest="test_bool",default=False,help="Invoke model testing")
	parser.add_option("--cool",action="store_true",dest="cool",default=False,help="Cool Learning Rate")
	parser.add_option("--case",help="classification case",dest="case",type=int,default=0)
	parser.add_option("--epochs",help="epochs",dest="epochs",type=int,default=1)
	parser.add_option("--batch-size",help="batch size",dest="batch_size",type=int,default=32)
	parser.add_option("--home-dir",help="Keras Directory, if provided will use as default keras path",dest="home_dir",type=str,default='%s/Work/Python/Git_Folder/keras'%HOME)
	parser.add_option("--save-dir",help="Save Directory",dest="save_path",type=str,default=None)
	parser.add_option("--memo",help="Memo",dest="base_memo",type=str,default=None)
	parser.add_option("--model-file",help="Trained Model Pickle File",dest="weight_file",type=str,default=None)
	parser.add_option("--seed", action="store_true",dest="seed",default=False,help="Random Seed")
	parser.add_option("--learning-rate",help="learning rate",dest="learning_rate",type=float,default=0.01)
	parser.add_option("--window-size",help="length of temporal depth for training",dest="window_size",type=int,default=1)

	(opts,args)=parser.parse_args()

	save_path=opts.save_path
	seed=opts.seed
	memo=opts.base_memo
	epochs=opts.epochs
	weight_file=opts.weight_file
	stateful_bool=True

	if os.path.isdir(opts.home_dir):
		sys.path.append(opts.home_dir)
	
	import keras.backend as K
	from keras.utils.np_utils import to_categorical
	from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
	from keras.callbacks import LearningRateScheduler,ModelCheckpoint
	from keras import callbacks
	
	sys.path.append(file_path)

	import eeg_classification.helper_functions as hf
	reload(hf)
	import eeg_classification.dataprovider as dp
	reload(dp)

	np.random.seed(100)
	
	learning_rate=opts.learning_rate
	cooling_boo1=opts.cool
	
	## Read the raw data and prep it for training with RNNs..
	## preping involves splitting data into smaller window sizes and setting the appropriate batch-size
	x,y,classes=dp.generate_classifier_data(case=opts.case,split_factor=50,seed=seed) ## generate train/test split
	X_train=np.array([x['train']]).transpose(1,2,0);y_train=to_categorical(y['train'],classes)
	X_test=np.array([x['test']]).transpose(1,2,0);y_test=to_categorical(y['test'],classes)
	#X_train=X_train/X_train.max(); X_test=X_test/X_train.max()
	train_samples=X_train.shape[0] ## original training sample
	test_samples=X_test.shape[0] ## original testing samples
	original_temporal_length=X_train.shape[1]
	
	if stateful_bool:
		window_length=opts.window_size ## temporal window size of sample to be trained
		skip_length=opts.window_size ## overlap window size	
		X_train,y_train=hf.prepare_sequences(X_train,y_train,window_length,skip_length) ## reformat data into smaller temporal-bins
		X_test,y_test=hf.prepare_sequences(X_test,y_test,window_length,skip_length)
		n_data_samples=X_train.shape[0] ## modified data sample size
		n_elem_per_sample=int(np.ceil(1.0*(original_temporal_length-window_length+1)/skip_length)) ## number of sub-samples that make up original data-sample
		batch_size=n_elem_per_sample ## set to number of sub-samples.. each batch should comprise of a single data-sample
		datagen=dp.DataGenerator()
		train_imggen=datagen.flow(X_train,y_train,batch_size)	
		test_imggen=datagen.flow(X_test,y_test,batch_size)
		
		print 'Batch_Size:',n_elem_per_sample

	print X_train.shape,y_train.shape
	class_weight=[]
	for d in np.unique(y['train']):
		class_weight.append(1.0*np.sum(y['train']==d)/y_train.shape[0])
		print 'number of examples in training set for class %d ==> %d'%(d,np.sum(y['train']==d))
		print 'number of examples in test set for class %d ==> %d'%(d,np.sum(y['test']==d))

	total_epochs = opts.epochs
	if cooling_boo1 and total_epochs<3:
		total_epochs=3
	

	opt=Adam(lr=learning_rate,clipvalue = 10.)
	model=hf.simple_recurrent_encoder_decoder_model(T=X_train.shape[1],D=X_train.shape[2],classes=classes,stateful=stateful_bool,batch_size=batch_size)
	#model=hf.simple_recurrent_model(T=X_train.shape[1],D=X_train.shape[2],classes=classes,stateful=stateful_bool,batch_size=batch_size)
	model.compile(loss='categorical_crossentropy', optimizer = opt,metrics=['accuracy'])
	model.summary()

	### set model hidden state manually
	# h1 = K.variable(value=np.zeros((51, 100)))
	# h2=K.variable(value=np.zeros((51, 100)))
	# model.layers[1].states[0] = h1
	# model.layers[3].states[0] = h2

	if weight_file!=None:
		print(weight_file)
		model.load_weights(weight_file)

	ts=time.time()
	if opts.train_bool:
		print 'Begin Training...'
		if cooling_boo1:
			train_scores,test_scores=hf.train_stateful_rnn_batch(model,class_weight,total_epochs/3,train_imggen,train_samples,test_imggen,test_samples,test_freq=2)
			print 'Cooling Learning Rate by factor of 10, continue training....'
			model.optimizer.lr.set_value(opts.learning_rate/10)
			tr_sc1,te_sc1=hf.train_stateful_rnn_batch(model,class_weight,total_epochs/3,train_imggen,train_samples,test_imggen,test_samples,test_freq=2)
			print 'Cooling Learning Rate by factor of 100, continue training....'
			model.optimizer.lr.set_value(opts.learning_rate/100)
			tr_sc2,te_sc2=hf.train_stateful_rnn_batch(model,class_weight,total_epochs/3,train_imggen,train_samples,test_imggen,test_samples,test_freq=2)	
			f_epoch=train_scores['epoch'][-1]+1
			train_scores['epoch'].extend(list(np.array(tr_sc1['epoch'])+f_epoch))
			train_scores['err'].extend(tr_sc1['err'])
			train_scores['acc'].extend(tr_sc1['acc'])

			train_scores['epoch'].extend(list(np.array(tr_sc2['epoch'])+2*f_epoch))
			train_scores['err'].extend(tr_sc2['err'])
			train_scores['acc'].extend(tr_sc2['acc'])

			test_scores['epoch'].extend(list(np.array(te_sc1['epoch'])+f_epoch))
			test_scores['err'].extend(te_sc1['err'])
			test_scores['acc'].extend(te_sc1['acc'])

			test_scores['epoch'].extend(list(np.array(tr_sc2['epoch'])+2*f_epoch))
			test_scores['err'].extend(tr_sc2['err'])
			test_scores['acc'].extend(tr_sc2['acc'])


		else:
			train_scores,test_scores=hf.train_stateful_rnn_batch(model,class_weight,total_epochs,train_imggen,train_samples,test_imggen,test_samples,test_freq=1)
		
		print 'Performance on test-set after completion of training....'
		agg_loss,agg_acc,total_score=hf.aggregrate_prediction_using_test_on_batch(model,test_imggen,test_samples)
		escore,dscore=hf.compute_early_seizure_detection_scores(total_score,test_samples,n_elem_per_sample)
		print '\nAccuracy on test-set: %.3f percent'%(100*agg_acc)
		print 'loss on test-set: %.3f'%(agg_loss)	
		

		if save_path!=None and memo !=None:
			pkl_file='%s/%s.pkl'%(save_path,memo)
			o=open(pkl_file,'wb')
			pickle.dump([train_scores,test_scores,escore],o)
			o.close()
			model_file='%s/%s.hdf5'%(save_path,memo)
			model.save_weights(model_file)

	if opts.test_bool:
		agg_loss,agg_acc,total_score=hf.aggregrate_prediction_using_test_on_batch(model,test_imggen,test_samples)
		escore,dscore=hf.compute_early_seizure_detection_scores(total_score,test_samples,n_elem_per_sample)
		print '\nAccuracy on test-set: %.3f percent'%(100*agg_acc)
		print 'loss on test-set: %.3f'%(agg_loss)
		if save_path!=None and memo!=None:
			pkl_file='%s/%s.pkl'%(save_path,memo)
			o=open(pkl_file,'wb')
			pickle.dump([agg_loss,agg_acc,escore,dscore],o)
			o.close()