import numpy as np
from keras.utils import np_utils
import sys,os
import glob
HOME=os.environ['HOME']
sys.path.append('%s/Work/Python/Git_Folder/sachin_projects/emd'%HOME)
import matplotlib
matplotlib.use('TKAgg')
import pylab as py
import cv2
import threading
####  STFT_Data path #####
#'%s/Work/DataSets/Epilepsy_Bonn/STFT_Data.pkl'%HOME

#py.savefig(fileName, transparent=True, bbox_inches='tight',pad_inches=0)

def SaveFigureAsImage(fileName,fig=None,**kwargs):
    ''' Save a Matplotlib figure as an image without borders or frames.
       Args:
            fileName (str): String that ends in .png etc.
 
            fig (Matplotlib figure instance): figure you want to save as the image
        Keyword Args:
            orig_size (tuple): width, height of the original image used to maintain 
            aspect ratio.
    '''
    fig_size = fig.get_size_inches()
    w,h = fig_size[0], fig_size[1]
    fig.patch.set_alpha(0)
    if kwargs.has_key('orig_size'): # Aspect ratio scaling if required
        w,h = kwargs['orig_size']
        w2,h2 = fig_size[0],fig_size[1]
        fig.set_size_inches([(w2/w)*w,(w2/w)*h])
        fig.set_dpi((w2/w)*fig.get_dpi())
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    py.axis('off')
    py.xlim(0,h); py.ylim(w,0)
    fig.savefig(fileName, transparent=True, bbox_inches='tight', \
                        pad_inches=0)


def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf


### Code base for low pass filtering input signal
from scipy.signal import butter, lfilter, freqz
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

### Example use to plot the filter and spectogram ####
#b, a = ehf.butter_lowpass(40, 173, 6)
#w, h = freqz(b, a, worN=8000)
#py.plot(0.5*173*w/np.pi, np.abs(h), 'b')
#py.figure();py.specgram(X_train[2,:,0], NFFT=256, Fs=173)



import emd.emd as emd
def detrend_data(x,nIMF=3):
    imfs=emd.emd(x,nIMF)
    xd=x-imfs[nIMF-1]
    return xd,imfs

def threshold_crossings(x,th=0):
    index=[i for i,y in enumerate(x[0:-1]) if x[i]<th and x[i+1]>=th]
    return index


def save_stft_img(x,outfile='tmp.png'):
    fig=py.figure()
    Pxx, freqs, bins, im=py.specgram(x, NFFT=256, Fs=173)
    py.clim([0,20])
    py.ylim([0,40])
    py.xlim([0,23])
    ax=fig.add_subplot(1,1,1)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    py.savefig(outfile, bbox_inches=extent)     
    py.close('all')    

### Compile the Epilepsy Bonn Data into one file
def compile_bonn_data(convert_to_stft=False,split_factor=80,data_dir='%s/Work/DataSets/Epilepsy_Bonn'%HOME):
    ### If convert_to_stft is true, the code will dump the STFT image data to file
    class_dict={'Z':0,'O':1,'N':2,'F':3,'S':4}
    count_dict={'Z':0,'O':0,'N':0,'F':0,'S':0}
    files=glob.glob('%s/Txt_Data/*'%data_dir)
    np.random.shuffle(files)
    X={'train':[],'test':[]};Y={'train':[],'test':[]}
    for f in files:
        if 'pkl' not in f:
            if 'png' not in f:
                f_str=f.split('/')[-1]
                data_str=open(f).readlines()
                x_data=[float(d.split('\r')[0]) for d in data_str]
                x_data=1.0*np.array(x_data)/2**11   ## fixed data normalization wrt AD-bit resolution
                x_data=butter_lowpass_filter(x_data,40,173,10)
                if convert_to_stft:
                    if 'txt' in f:
                        outfile=data_dir+'/Png_Data/'+f_str.replace('.txt','.png')
                    if 'TXT' in f:
                        outfile=data_dir+'/Png_Data/'+f_str.replace('.TXT','.png')
                    print outfile
                    fig=py.figure()
                    Pxx, freqs, bins, im=py.specgram(x_data, NFFT=256, Fs=173)
                    py.clim([0,20])
                    py.ylim([0,40])
                    py.xlim([0,23])
                    ax=fig.add_subplot(1,1,1)
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    py.savefig(outfile, bbox_inches=extent)     
                    py.close('all')
                    #img=cv2.imread('/Users/talathi1/Desktop/tmp.png')
                    #x_data=img.copy()     
                y_data=class_dict[f_str[0]]
                if count_dict[f_str[0]]<split_factor:
                    X['train'].append(x_data)
                    Y['train'].append(y_data)
                else:
                    X['test'].append(x_data)
                    Y['test'].append(y_data)
                count_dict[f_str[0]]+=1
    return X,Y

def mapping(x,dic):
    #print dic
    for k in dic.keys():
        x[np.where(x==k)[0]]=dic[k]
    return x

def generate_classifier_data(convert_to_stft=False,case=0,split_factor=80,seed=False):
    ##### There are 7 cases to consider for division of classes ########

    #case=0==> Classes: {Healthy, Inter-Ictal, Ictal}={(A,B),(C,D),E}
    #case=1==> Classes: {Non-Seizure, Seizure}={(A,B,C,D),E}
    #case=2==> Classes: {Healthy, Seizure}={(A,E}
    #case=3==> Classes: {Healthy, Inter-Ictal, Ictal}={(A,D,E}
    #case=4==> Classes: {Healthy, Inter-Ictal, Ictal}={(A,C,E}
    #case=5==> Classes: {Inter-Ictal, Ictal}={D,E}
    #case=6==> Classes: {Inter-Ictal, Ictal}={C,E}
    if seed:
        np.random.seed(100)
    else:
        np.random.seed(np.random.randint(10000))
    x,y=compile_bonn_data(convert_to_stft=convert_to_stft,split_factor=split_factor)
    x['train']=np.array(x['train']); x['test']=np.array(x['test'])
    y['train']=np.array(y['train']); y['test']=np.array(y['test'])

    N_train=x['train'].shape[0]; N_test=x['test'].shape[0]

    index_train=np.arange(0,N_train)
    np.random.shuffle(index_train)
    x['train']=x['train'][index_train,:]; y['train']=y['train'][index_train]

    index_test=np.arange(0,N_test)
    np.random.shuffle(index_test)
    x['test']=x['test'][index_test,:]; y['test']=y['test'][index_test]

    
    if case==0:
        dic={0:0,1:0,2:1,3:1,4:2}
        classes=3
    elif case==1:
        dic={0:0,1:0,2:0,3:0,4:1}
        classes=3
    elif case==2:
        dic={0:0,4:1}
        classes=2
        ind=np.where((y['train']==0) | (y['train']==4))[0]
        x['train']=x['train'][ind,:];y['train']=y['train'][ind]
        ind=np.where((y['test']==0) | (y['test']==4))[0]
        x['test']=x['test'][ind,:];y['test']=y['test'][ind]
    elif case==2:
        dic={0:0,4:1}
        classes=2
        ind=np.where((y['train']==0) | (y['train']==4))[0]
        x['train']=x['train'][ind,:];y['train']=y['train'][ind]
        ind=np.where((y['test']==0) | (y['test']==4))[0]
        x['test']=x['test'][ind,:];y['test']=y['test'][ind]
    elif case==3:
        dic={0:0,3:1,4:2}
        classes=3
        ind=np.where((y['train']==0) | (y['train']==3) | (y['train']==4))[0]
        x['train']=x['train'][ind,:];y['train']=y['train'][ind]
        ind=np.where((y['test']==0) | (y['test']==3)| (y['test']==4))[0]
        x['test']=x['test'][ind,:];y['test']=y['test'][ind]
    elif case==4:
        dic={0:0,2:1,4:2}
        classes=3
        ind=np.where((y['train']==0) | (y['train']==2) | (y['train']==4))[0]
        x['train']=x['train'][ind,:];y['train']=y['train'][ind]
        ind=np.where((y['test']==0) | (y['test']==4)| (y['test']==2))[0]
        x['test']=x['test'][ind,:];y['test']=y['test'][ind]
    elif case==5:
        classes=2
        dic={3:0,4:1}
        ind=np.where((y['train']==3) | (y['train']==4))[0]
        x['train']=x['train'][ind,:];y['train']=y['train'][ind]
        ind=np.where((y['test']==3) | (y['test']==4))[0]
        x['test']=x['test'][ind,:];y['test']=y['test'][ind]

    else:
        dic={2:0,4:1}
        classes=2
        ind=np.where((y['train']==2) | (y['train']==4))[0]
        x['train']=x['train'][ind,:];y['train']=y['train'][ind]
        ind=np.where((y['test']==2) | (y['test']==4))[0]
        x['test']=x['test'][ind,:];y['test']=y['test'][ind]

   
    y['train']=mapping(y['train'],dic)
    y['test']=mapping(y['test'],dic)

    return x,y,classes

def create_dataset(dataset, look_back=1,look_forward=1):
    ## input is np.array of dim T,D
    #output is np.array X: N,look_back,D and Y: N,D
    # where N=T-look_back-1
    assert(look_back>=look_forward)
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-(look_forward-1)-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back:i+look_back+look_forward, :])
    dataX=np.array(dataX)
    dataY=np.array(dataY)    
    if look_back-look_forward>0:
        dataY_mod=np.zeros((dataY.shape[0],dataX.shape[1],dataY.shape[2]))
        dataY_mod[:,0:dataY.shape[1],:]=dataY
    else:
        dataY_mod=dataY
    return dataX, dataY_mod

def generate_forecast(model,x,prediction_length=10):
    #To be use when rnn model is trained to predict a single value output
    #x: shape 1,T,D
    #y_pred: shape 1,D ( representing prediction at time T+1)
    (N,T,D)=x.shape
    x_data=x[0,:,:]
    for i in range(prediction_length):
        y_pred=model.predict(x[0:1,:,:])
        x_data=np.vstack([x_data,y_pred])
        #T,D=x_data.shape
        #x=x_data.reshape(1,T,D)
        x=x.reshape(T,D) ## assume N=1 ... i.e. one sample
        #print 'data before prediction:\n',x
        x[0:T-1,:]=x[1:T,:]
        x[T-1:T,:]=y_pred
        x=x.reshape(1,T,D)
        print 'data after appending prediction\n',x    
    return x_data

def generate_timedistributed_forecast(model,x,prediction_length=10):
    ## to be used when rnn is used for sequence to sequence mapping
    N,T,D=x.shape
    x_data=x[0,:,:]
    for i in range(prediction_length):
        y_pred=model.predict(x[0:1,:,:],batch_size=1)
        yf=y_pred[:,0,:]
        print 'prediction:',yf
        x_data=np.vstack([x_data,yf])
        
        #x=x.reshape(T,D) ## assume N=1 ... i.e. one sample
        print 'data before prediction:\n',x
        x[0,0:T-1,:]=x[0,1:T,:]
        x[0,T-1:T,:]=yf
        #x=x.reshape(1,T,D)
        print 'data after appending prediction\n',x    
    return x_data    

############# Define Data Generators ################

class DataGenerator(object):
    '''Generate minibatches with
    realtime data augmentation.
    '''
    def __init__(self,corruption_level=0.):

        self.__dict__.update(locals())
        self.p=corruption_level
        self.lock = threading.Lock()

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        b = 0
        total_b = 0
        while 1:
            if b == 0:
                if seed is not None:
                    np.random.seed(seed + total_b)

                if shuffle:
                    index_array = np.random.permutation(N)
                else:
                    index_array = np.arange(N)

            current_index = (b * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
            else:
                current_batch_size = N - current_index

            if current_batch_size == batch_size:
                b += 1
            else:
                b=0
                #b=None
            
            #if current_index + current_batch_size==N:
            #   b=None
            total_b += 1
            yield index_array[current_index: current_index + current_batch_size], current_index, current_batch_size
            #if b==None:
            #    return

    def flow(self, X, y, batch_size=32, shuffle=False, seed=None):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.flow_generator = self._flow_index(X.shape[0], batch_size, shuffle, seed)
        return self

    def __iter__(self):
        # needed if we want to do something like for x,y in data_gen.flow(...):
        return self

    def next(self):
        # for python 2.x
        # Keep under lock only the mechainsem which advance the indexing of each batch
        # see # http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.flow_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        bX = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            #x = self.insertnoise(x,corruption_level=self.p)
            bX[i] = x
        bY = self.y[index_array]
        return bX, bY

    def __next__(self):
        # for python 3.x
        return self.next()

    def insertnoise(self,x,corruption_level=0.5):
        return np.random.binomial(1,1-corruption_level,x.shape)*x

