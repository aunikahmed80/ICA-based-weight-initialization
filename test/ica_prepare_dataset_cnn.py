



import gzip

import gc
import numpy as np
from PIL import Image

import pickle as cPickle

#import theano
from sklearn import preprocessing
from sklearn.decomposition import FastICA
#from theano.tensor.signal import downsample

import my_mnist_helper
from utils import tile_raster_images

#from theano import tensor as T

#from theano.tensor.nnet import conv as conv_nnet
import scipy.io as sio


def show_img(raster):
    image = Image.fromarray(tile_raster_images(
        X=raster,
        img_shape=(28, 28), tile_shape=(9, 16),
        tile_spacing=(2, 2)))
    image.show()


def get_digit_set(digit, type = 'train'):
    f = gzip.open('../../book_code/data/mnist.pkl.gz', 'rb')
    train_set, test_set, valid_set = cPickle.load(f)
    if type == 'train':
        x,y = train_set
    elif type =='test':
        x,y = test_set
    elif type == 'valid':
        x,y = valid_set

    if digit =='all':
        return (x,y)
    x= x[y==digit]
    return (x,y)




def get_digit_set_for_feature_extraction():
    f = gzip.open('../../data/mnist.pkl.gz', 'rb')
    train_set, test_set, valid_set = cPickle.load(f)
    x,y = train_set

    x1 = x[y==5 ]
    x2 = x[y==3]
    x = np.append(x1,x2,axis=0)
    return (x,y)


def get_unmixed_components(n_comp=2):
    ica_observations, ica_observations_y = get_digit_set_for_feature_extraction()
    # Compute ICA
    ica_observations = ica_observations - np.mean(ica_observations,axis=0)
    ica = FastICA(n_components=n_comp)
    S = ica.fit_transform(ica_observations.T)  # Reconstruct signals
    return S.T


def get_extreme(s):

    for i,m in enumerate(s):
        d = np.ma.array(m)
        d_mean = np.mean(d)
        d_std = np.std(d)
        d =np.ma.masked_inside(d, d_mean - d_std, d_mean + d_std)
        d = d.filled(0)
        s[i] = d

    return s


def get_basis_features(feature_per_dig):
    s=[]
    s = get_unmixed_components(feature_per_dig)

    #s= np.reshape(s,[10*feature_per_dig,-1])
    #s = get_extreme(s)
    s_temp = - s
    #s = np.append(s,s_temp,axis=0)

    show_img(s)

    s = np.reshape(s, [-1,28,28])

    return s


def get_conv_feature(s, inp):
    inp = -1+inp*2 #scaling between (-1,1)

    x =  theano.shared(
            np.reshape(inp,[-1,1,28,28]), borrow=True
    )

    X = T.tensor4()
    W = T.tensor4()
    basis_feature_shape = np.shape(s)
    s = np.reshape(s, [-1,1,basis_feature_shape[1],basis_feature_shape[2]])
    Ws =  theano.shared(
        np.asarray(
                s, dtype= theano.model_config.floatX
        ),
        borrow =True
    )
    conv_out = theano.tensor.nnet.conv2d(X,W,border_mode=(3),filter_flip=False)

    pooled_out = downsample.max_pool_2d(
        input=conv_out,
        ds=[7,7],
        ignore_border=True
    )

    c2 =  theano.function([], pooled_out, givens={W:Ws, X:x})
    c = c2()
    print(np.shape(c))
    c = np.reshape(c,[c.shape[0],c.shape[1]])
    gc.collect()
    #c_normalized = preprocessing.scale(c)
    return c


def digit_set_with_distortion(undistored_digit_set_x , y):
    kernel_dim = 13
    sigma = 8
    digit_dim = 28
    digit_set_shape = np.shape(undistored_digit_set_x)

    y_distored_set = np.zeros(2 * digit_set_shape[0])
    distorted_digit_set_x = np.zeros([2 * digit_set_shape[0],digit_set_shape[1]]).astype('float32')

    kernel = my_mnist_helper.create_2d_gaussian(kernel_dim, sigma)
    for i,digit in enumerate(undistored_digit_set_x):
        distorted_digit,elastic_field = my_mnist_helper.elastic_transform(np.reshape(digit, (digit_dim, digit_dim)),
                                                                    kernel,
                                                                    alpha=20,
                                                                     negated=True)
        #print distorted_digit.dtype
        distorted_digit_set_x[2*i] = digit
        distorted_digit_set_x[2*i+1] = distorted_digit.flatten()
        y_distored_set[2*i] = y[i]
        y_distored_set[2*i+1] = y[i]
    return distorted_digit_set_x, y_distored_set




def load_ica_conv_dataset_after_pooling(n_comp):


    s = get_basis_features(n_comp)
    print(np.shape(s))
    indx =403
    test_set_x , test_set_y = get_digit_set('all','test')
    test_set_feature = get_conv_feature(s, test_set_x[indx])
    print(test_set_feature)

    show_img(np.reshape(test_set_x[indx],[1,-1]))



def show_reconstraction(n_comp=10):
    ica_observations, ica_observations_y = get_digit_set(5)
    # Compute ICA
    ica = FastICA(n_components=n_comp)
    S = ica.fit_transform(ica_observations.T)  # Reconstruct signals
    x = np.dot(S, ica.mixing_.T) #+ ica.mean_
    print(S[0, 0])
    S[0,0] = 0
    x1 = np.dot(S, ica.mixing_.T)

    show_img((x.T[0],x1.T[0]))
    return S.T


#show_reconstraction()
get_basis_features(4)
#load_ica_conv_dataset_after_pooling(10)