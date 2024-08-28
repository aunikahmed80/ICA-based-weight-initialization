



import gzip

import gc
from PyQt4 import QtGui

import numpy as np
from PIL import Image

import pickle as cPickle


from sklearn import preprocessing
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import normalize
#from theano.tensor.signal import downsample

#from my_mnist_helper import ElasticDistortion
from utils import tile_raster_images

#from theano import tensor as T

from theano.tensor.nnet import conv as conv_nnet
import scipy.io as sio


def show_img(raster, img_shape = (28,28), tile_shape = (9,16)):
    image = Image.fromarray(tile_raster_images(
        X=raster,
        img_shape=img_shape, tile_shape=tile_shape,
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
    ica = FastICA(n_components=n_comp)
    S = ica.fit_transform(ica_observations.T)   # Reconstruct signals
    return S.T
#


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

    show_img(s, (28,28),(10,20))

    s = np.reshape(s, [-1,28,28])
    return s




def get_non_rotated_basis_features_from_file(filename = '../cnn_using_ica_weight_initialization/ica_basis_feature_0.91_0.93'):
    f = open(filename)
    s = cPickle.load(f)
    f.close()
    return s


def gram_matrix(x):
    g = np.dot(x, x.T)
    return g

def plot_3d(g):
    import numpy
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Set up grid and test data
    nx, ny = 200,200
    x = range(nx)
    y = range(ny)

    data = g

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, data)

    plt.show()

def show_in_table(g):
    datatable = QtGui.QTableWidget()
    for i in range(200):
        for j in range(200):
            datatable.setItem(i, j, QtGui.QTableWidgetItem(str(g[i,j])))


def test_feature_closeness(feature):
    s = np.reshape(feature, [feature.shape[0], -1])
    s = normalize(s)
    g = np.dot(s, s.T)

    g = g[np.where(np.abs(g) > .6)]
    l_6 = len(g) - 200

    g = g[np.where(np.abs(g) > .7)]
    l_7 = len(g) -200

    g = g[np.where(np.abs(g) > .8)]
    l_8 = len(g) -200

    print '>.6:', l_6, '\t>.7:', l_7,'\t>.8:', l_8


if __name__ == '__main__':
    s = get_non_rotated_basis_features_from_file()
    test_feature_closeness(s)

    #plot_3d(g)
    #show_in_table(g)
    # #
    # s = np.asarray([[1.,2],[1,-1]])
    # s = normalize(s)

    #print np.min(g)#gram_matrix(s)