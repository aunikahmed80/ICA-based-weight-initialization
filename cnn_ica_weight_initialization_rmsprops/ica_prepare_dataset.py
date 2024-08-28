



import gzip

import gc
import numpy as np
from PIL import Image

import cPickle

import theano
from sklearn import preprocessing
from sklearn.decomposition import FastICA, PCA
from theano.tensor.signal import downsample

#from my_mnist_helper import ElasticDistortion
from utils import tile_raster_images

from theano import tensor as T

from theano.tensor.nnet import conv as conv_nnet
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


def get_unmixed_components(digit, n_comp=10):
    ica_observations, ica_observations_y = get_digit_set(digit)
    # pca = PCA(n_components=64)
    # x = pca.fit_transform(ica_observations)
    # ica_observations = pca.inverse_transform(x)
    # Compute ICA
    rns = np.random.RandomState(1234)
    ica = FastICA(n_components=n_comp)
    S = ica.fit_transform(ica_observations.T)   # Reconstruct signals
    return S.T
#
# pca = decomposition.PCA(n_components=3)
# pca.fit(X)
# X = pca.transform(X)

def get_extreme(s):

    for i,m in enumerate(s):
        d = np.ma.array(m)
        d_mean = np.mean(d)
        d_std = np.std(d)
        d =np.ma.masked_inside(d, d_mean - d_std, d_mean + d_std)
        d = d.filled(0)
        s[i] = d

    return s
def get_scaled_feature(s):
    s_min = np.min(s)
    s_max = np.max(s)
    r = s_max - s_min
    s = (s - s_min) / r
    return s

def get_basis_features(feature_per_dig):
    s=[]
    for i in xrange(10):
        x = get_unmixed_components(i,feature_per_dig)
        s.append( x)
    s= np.reshape(s,[10*feature_per_dig,-1])
    s = get_extreme(s)

    show_img(s)

    s = np.reshape(s, [-1,28,28])
    return s




def save_basis_feature(s):
    f = open('ica_basis_feature_temp','wb')
    cPickle.dump(s,f , -1)
    f.close()


def get_non_rotated_basis_features_from_file(filename = 'ica_basis_feature_0.91_0.93'):
    f = open(filename)
    s = cPickle.load(f)
    f.close()
    return s


#valid_set_x, valid_set_y = get_digit_set('all','valid')
#digit_set_with_distortion(valid_set_x, valid_set_y)

if __name__ == '__main__':
    s = get_basis_features(20)
    s_min = np.min(s)
    s_max = np.max(s)
    r = s_max - s_min
    s = (s - s_min) / r
    print np.min(s),'\t',np.max(s)