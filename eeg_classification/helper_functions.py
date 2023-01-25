## import segnet_keras
from __future__ import absolute_import
import os,sys
import theano
import matplotlib
if 'MACOSX' in matplotlib.get_backend().upper():
  matplotlib.use('TKAgg')
import pylab as py
py.ion() ## Turn on plot visualization
import gzip,pickle
import numpy as np
from PIL import Image
import cv2
import keras.backend as K
K.set_image_dim_ordering('th')
from keras.layers import Input, merge, TimeDistributed,LSTM,GRU,RepeatVector
from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape, Permute
from keras.initializers import normal, identity, he_normal,glorot_normal,glorot_uniform,he_uniform
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import threading
HOME=os.environ['HOME']
from keras.callbacks import Callback

## define call back for state
class ResetStatesCallback(Callback):
    def __init__(self,max_len=1):
        self.counter = 0
        self.max_len=max_len

    def on_batch_begin(self, batch, logs={}):
        if self.counter % self.max_len == 0:
            self.model.reset_states()
        self.counter += 1

## convert long sequence into sub-sequences
def prepare_sequences(x_train, y_train, window_length,skip=1):
    windows = []
    windows_y = []
    for i, sequence in enumerate(x_train):
        len_seq = len(sequence)
        for window_start in range(0, len_seq - window_length + 1,skip):
            window_end = window_start + window_length
            window = sequence[window_start:window_end]
            windows.append(window)
            windows_y.append(y_train[i])
    return np.array(windows), np.array(windows_y)

## generic utility function to plot figure
def plotfig(datalist):
	if not type(datalist)==list:
		print 'Input is datalist'
		sys.exit(0)
	
	N=len(datalist)
	fig=py.figure()
	ax=fig.add_subplot(111)
	for i in range(N):
		ax.plot(datalist[i])
		ax.hold('on')
	fig.canvas.show()


## Get activaitons
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations

def simple_recurrent_model(T=1,D=1,classes=3,stateful=False,batch_size=1):
    if stateful:
    	inp = Input(batch_shape=(batch_size,T,D))
    else:
    	inp = Input(shape=(T,D))
    encoder=LSTM(10,name='lstm_0',return_sequences=True, stateful=stateful)(inp)
    encoder=LSTM(10,name='lstm_1',return_sequences=True, stateful=stateful)(encoder)    
    encoder=LSTM(10,name='lstm_2',return_sequences=False, stateful=stateful)(encoder)
    encoder=Dense(classes,name='dense_0',activation='softmax',kernel_initializer='glorot_normal')(encoder)
    model=Model(inputs=inp,outputs=encoder)
    return model

def simple_recurrent_encoder_decoder_model(T=1,D=1,classes=3,stateful=False,batch_size=1):
	#Encoder-Decoder Design for Seq-To-Seq Learning
	np.random.seed(100)
	input_shape=(batch_size,T,D)
	inp = Input(batch_shape=input_shape)
	encoder=GRU(100,name='gru_0',stateful=stateful,return_sequences=True)(inp)
	encoder=TimeDistributed(Dense(100,name='dense_0',activation='linear',kernel_initializer='glorot_normal'))(encoder)
	#encoder=RepeatVector(T)(encoder)
	encoder=GRU(100,name='gru_1',return_sequences=False,batch_input_shape=(batch_size, T, D),stateful=stateful)(encoder)
	encoder=Dense(classes,name='dense_1',activation='softmax',kernel_initializer='glorot_normal')(encoder)
	model=Model(inputs=inp,outputs=encoder)
	return model

def train_stateful_rnn(model,X_train,y_train,X_test,y_test,epochs,or_temporal_length,window_length,skip_length):
	#n_elem_per_sample=ceil((temporal_depth_original_sequence-sub_temporal_window+1)/skip_window)
	n_train_samples=X_train.shape[0]
	n_test_samples=X_test.shape[0]
	n_elem_per_sample=int(np.ceil(1.0*(or_temporal_length-window_length+1)/skip_length))
	
	for epoch in range(epochs):
		mean_tr_acc = []
		mean_tr_loss = []
		for i in range(0,n_train_samples,n_elem_per_sample):
			
			y_true = y_train[i:i+1]
			sub_ta=[]
			sub_tl=[]
			for j in range(n_elem_per_sample):
				#print X_train[i+j:i+j+1].shape,y_true.shape
				tr_loss, tr_acc = model.train_on_batch(X_train[i+j:i+j+1],y_true)
				sub_ta.append(tr_acc)
				sub_tl.append(tr_loss)
			mean_tr_acc.append(np.mean(sub_ta))
			mean_tr_loss.append(np.mean(sub_tl))
			model.reset_states()
		print np.array(mean_tr_acc).shape	
		print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
		print('loss training = {}'.format(np.mean(mean_tr_loss)))
		print('___________________________________')

	
		mean_te_acc = []
		mean_te_loss = []
		for i in range(0,n_test_samples,n_elem_per_sample):
			y_true = y_test[i:i+1]
			sub_ta=[]
			sub_tl=[]
			for j in range(n_elem_per_sample):
				te_loss, te_acc = model.test_on_batch(X_test[i+j:i+j+1],y_true)
				sub_ta.append(te_acc)
				sub_tl.append(te_loss)
			mean_te_acc.append(np.mean(te_acc))
			mean_te_loss.append(np.mean(te_loss))
			model.reset_states()
		print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
		print('loss testing = {}'.format(np.mean(mean_te_loss)))
		print('___________________________________')
	

def train_stateful_rnn_batch(model,class_weight,epochs,train_imggen,train_Iter,test_imggen,test_Iter,test_freq=1):
	train_scores={'epoch':[],'err':[],'acc':[]}
	test_scores={'epoch':[],'err':[],'acc':[]}
	
	for epoch in range(epochs):
		iter_loss=[];iter_acc=[]
		for _ in range(train_Iter):
			x,y=next(train_imggen)
			tr_loss,tr_acc=model.train_on_batch(x,y,class_weight=class_weight)
			iter_loss.append(tr_loss)
			iter_acc.append(tr_acc)
			model.reset_states()
		print 'For epoch %d...'%epoch, ' Training Loss:%.3f'%np.mean(iter_loss),' Training Acc:%.3f'%np.mean(iter_acc)
		train_scores['epoch'].append(epoch); train_scores['err'].append(np.mean(iter_loss));train_scores['acc'].append(np.mean(iter_acc))

		if epoch % test_freq==0:
			agg_loss,agg_acc,tscore=aggregrate_prediction_using_test_on_batch(model,test_imggen,test_Iter)
			#te_iter_loss=[];te_iter_acc=[]
			# for _ in range(test_Iter):
			# 	x,y=next(test_imggen)
			# 	te_loss,te_acc=model.test_on_batch(x,y)
			# 	te_iter_loss.append(te_loss)
			# 	te_iter_acc.append(te_acc)
			# 	model.reset_states()
			print '..............Validation Loss:%.3f'%agg_loss,' Validation Acc:%.3f'%agg_acc			
			test_scores['epoch'].append(epoch); test_scores['err'].append(agg_loss);test_scores['acc'].append(agg_acc)
	
	return train_scores,test_scores


### This is the right way to evaluate model performance using stateful rnns.. primarily because, model states are to be reset at end of each batch
def aggregrate_prediction_using_test_on_batch(model,test_imggen,test_Iter,mode='MEAN'):
	np.random.seed(100)
	agg_score=[]
	agg_loss=[]
	total_score=[]
	model.reset_states()
	count=0
	for _ in range(test_Iter):
		count+=1

		x,y=next(test_imggen)
		x_c=x.reshape(x.shape[0]*x.shape[1])
		test_loss,test_acc=model.test_on_batch(x,y)
		model.reset_states()
		#agg_score.append(test_acc>0.5)
		agg_loss.append(test_loss)
	## different way of computing scores:
		pred=model.predict(x,x.shape[0])
		model.reset_states()

		results=np.correlate(x_c,x_c,mode='Full')
		r=results[results.size/2:]
		ind=np.where(r<0)[0]

		total_score.append([np.argmax(pred,axis=1),np.argmax(y,axis=1)])
		if mode.upper()=='MEAN':
		#if len(ind)!=0:
			agg_score.append(1.0*sum(np.argmax(y,axis=1)==np.argmax(pred,axis=1))/x.shape[0]>0.5)
		if mode.upper()=='LAST':
		#else:
			agg_score.append(np.argmax(y[-1])==np.argmax(pred[-1]))
		
	
	return np.mean(agg_loss),1.0*sum(agg_score)/len(agg_score),total_score

### Code to count value in list that is repeated maximum number of times 
'''
for i in range(250):
    val=collections.Counter(ts[i][0])
    pred_label.append(max(val.iteritems(), key=operator.itemgetter(1))[0])
'''

## compute early seizure detection accuracy as function of number of sub-segments
#test_Iter=250
#n_subsegments=51
def compute_early_seizure_detection_scores(total_score,test_Iter,n_subsegments):
	scores=np.zeros((test_Iter,n_subsegments))
	full_score=[]
	for k in range(test_Iter):
		full_score.append(1.0*np.sum(total_score[k][0]==total_score[k][1])/(n_subsegments+1)>0.5)
		for i in range(n_subsegments):
			scores[k,i]=1.0*np.sum(total_score[k][0][0:i+1]==total_score[k][1][0:i+1])/(i+1)>0.5
	return scores,1.0*np.sum(full_score)/len(full_score)
