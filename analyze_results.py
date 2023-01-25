import numpy as np
import sys,os
import glob
import matplotlib
matplotlib.use('TKAgg')
import pylab as py
py.ion()


## setting for py
py.style.use('seaborn-white')
py.rcParams['font.family'] = 'serif'
py.rcParams['font.serif'] = 'Ubuntu'
py.rcParams['font.monospace'] = 'Ubuntu Mono'
py.rcParams['font.size'] = 10
py.rcParams['axes.labelsize'] = 10
py.rcParams['axes.labelweight'] = 'bold'
py.rcParams['axes.titlesize'] = 10
py.rcParams['xtick.labelsize'] = 8
py.rcParams['ytick.labelsize'] = 8
py.rcParams['legend.fontsize'] = 10
py.rcParams['figure.titlesize'] = 12

### Code base to generate figures ###########
### Plot the results for ML training
img_x=[32,64,128,256]
img_y=[40,80,160,320]
Results={'40':[],'80':[],'160':[],'320':[]}

## Note files is the list of all pkl files with training and testing accuracy scores
HOME=os.environ['HOME']
Model_Dir='%s/Work/Python/Models/keras_models/conveeg_models'%HOME
files=glob.glob('%s/*.pkl'%Model_Dir)
case=5
for f in files:
	if int(f.split('.pkl')[0].split('_simple_')[0].split('_')[-1])==case:
		for sz in img_y:
			o=open(f)
			D=pickle.load(o)
			o.close()
			if str(sz) in f:
				Results[str(sz)].append(max(D['val_categorical_accuracy']))

data=np.zeros((20,4))
count=0
for sz in img_y:
	data[:,count]=np.array(Results[str(sz)])
	count+=1

fig, axes = py.subplots(nrows=1, ncols=1, figsize=(6, 6), sharey=True)
axes.boxplot(data, labels=labels,showmeans=True)
py.xlabel('Image Resolution (X-dimension)',fontsize=15)
py.ylabel('Classification accuracy on test-set',fontsize=15)


#### Plot raw data and the corresponding spectrum data
HOME=os.environ['HOME']
files=glob.glob('%s/Work/DataSets/Epilepsy_Bonn/Png_Data/*.png'%HOME) ## list of all png files 
txtfiles=glob.glob('%s/Work/DataSets/Epilepsy_Bonn/Txt_Data/*.*'%HOME)

sample_files=['Z081','O075','N036','F083','S091']

pngfiles=[]
for f in files:
	for s in sample_files:
		if s in f:
			pngfiles.append(f)
rawfiles=[]
for f in txtfiles:
	for s in sample_files:
		if s in f:
			rawfiles.append(f)



DataPair={}
for i in range(len(rawfiles)):
	ref=rawfiles[i].split('/')[-1].split('.')[0]
	data=open(rawfiles[i]).readlines()
	x_data=[float(d) for d in data]
	results=np.correlate(x_data,x_data,mode='Full')
	results_half=results[results.size/2:]
	fig=py.figure()
	Pxx, freqs, bins, im=py.specgram(x_data, NFFT=256, Fs=173)
	py.clim([0,20])
	py.ylim([0,87])
	py.xlim([0,23])
	ax=fig.add_subplot(1,1,1)
	extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
	outfile='%s/Desktop/spec.png'%HOME
	py.savefig(outfile, bbox_inches=extent)
	py.close('all')

	img=cv2.imread(outfile)
	img=Image.fromarray(img)
	b,g,r=img.split()
	img=Image.merge("RGB",(r,g,b))
	img_arr=np.array(img)	
	DataPair[ref]=[ref,np.array(x_data),img_arr,results_half]

 

a0.axis('tight')
a0.axis('off')
a1.axis('tight')
a1.axis('off')
a2.axis('tight')
a2.axis('off')
a3.axis('tight')
a3.axis('off')
a4.axis('tight')
a4.axis('off')
#a.axis('off')

ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
ax5.axis('off')
ax6.axis('off')
ax7.axis('off')
ax8.axis('off')
ax9.axis('off')
ax10.axis('off')
#ax10.axis('off')

### Generate performance curves
model_dir='/home/talathi1/Work/Python/Models/keras_models/rnneeg_models'
py.figure()
for i in range(10):
	res_file='%s/rnn_eeg_model_case0_w80_200epochs_correct_normalization_%d.pkl'%(model_dir,i)
	o=open(res_file)
	D=pickle.load(o)
	o.close()
	py.subplot(5,2,i+1)
	py.plot(D[0]['epoch'],D[0]['acc'],'b',linewidth=2)
	py.hold('on');py.plot(D[1]['epoch'],D[1]['acc'],'r',linewidth=2)

