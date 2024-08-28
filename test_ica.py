



import gzip

import numpy as np
from PIL import Image

import cPickle

import theano
from sklearn.decomposition import FastICA

from utils import tile_raster_images

from theano import tensor as T

import theano.tensor.signal.conv as conv_signal
from theano.tensor.nnet import conv as conv_nnet


def show_img(raster):
    image = Image.fromarray(tile_raster_images(
        X=raster,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(2, 2)))
    image.show()


def get_digit_set(digit, type = 'train'):
    f = gzip.open('../book_code/data/mnist.pkl.gz', 'rb')
    train_set, test_set, valid_set = cPickle.load(f)
    x,y = train_set

    x1 = x[y==8 ]
    x2 = x[y==9]
    x = np.append(x1,x2,axis=0)
    return (x,y)

#
# def get_digit_set(digit, type = 'train'):
#     f = gzip.open('../book_code/data/mnist.pkl.gz', 'rb')
#     train_set, test_set, valid_set = cPickle.load(f)
#     if type == 'train':
#         x,y = train_set
#     elif type =='test':
#         x,y = test_set
#     elif type == 'valid':
#         x,y = valid_set
#
#     if digit =='all':
#         return (x,y)
#     x= x[y==digit]
#     show_img(x[0:1])
#     return (x,y)
#

def get_unmixed_components(digit, n_comp=10):
    ica_observations, ica_observations_y = get_digit_set(digit)
    # Compute ICA
    #ica_observations = ica_observations - np.mean(ica_observations,axis=0)
    ica = FastICA(n_components=400)
    S = ica.fit_transform(ica_observations.T)  # Reconstruct signals
    #assert np.allclose(ica_observations.T, np.dot(S, ica.mixing_.T) + ica.mean_)
    print 'number of iteration take to converge ica: ',ica.n_iter_
    #assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)
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

def get_basis_features( feature_per_dig, digit = 'all'):
    s=[]
    if(digit == 'all'):
        for i in xrange(10):
            x = get_unmixed_components(i,feature_per_dig)
            s.append( x)
            s= np.reshape(s,[10*feature_per_dig,-1])
    else:
        s = get_unmixed_components(digit, feature_per_dig)


    s = get_extreme(s)

    show_img(s)

    s = np.reshape(s, [-1,28,28])
    s = [np.rot90(r,2) for r in s]

    return s



def get_conv_feature(s, x):

    x = theano.shared(
            np.reshape(x,[-1,1,28,28]), borrow=True
    )
    X = T.tensor4()
    s = np.reshape(s, [-1,1,28,28])
    #s = np.rot90(s,2)
    W = theano.shared(
        np.asarray(
                s, dtype=theano.model_config.floatX
        ),
        borrow =True
    )
    con = conv_nnet.conv2d(X,W,border_mode='valid')

    f = theano.function([],con, givens={W:W,X:x})
    c = f()
    c = np.reshape(c,[c.shape[0],c.shape[1]])

    return c


def show_conv_feature(digit, n_comp):


    s = get_basis_features(n_comp, digit)

    train_set_x, train_set_y = get_digit_set(digit,'train')
    train_set_feature = get_conv_feature(s, train_set_x[0])
    print 'train set feature shape', np.shape(train_set_feature)
    print train_set_feature

#get_basis_features(2,digit=6)


def test1():
    A = get_digit_set(4)[0]
    print np.shape(np.mean(A,1))
    #S = get_unmixed_components(digit=4, n_comp=20)

    # S = S.T
    # print np.shape(A), np.shape(S[:,0])
    #
    # X = np.linalg.lstsq(A, S[:,0])[0]
    # print np.shape(X)
    #


if __name__ == '__main__':
    test1()