



import gzip

import numpy as np
from PIL import Image

import cPickle

import theano
from sklearn.decomposition import FastICA

from utils import tile_raster_images
#from theano.tensor.nnet import conv
from theano import tensor as T

import theano.tensor.signal.conv as conv



def show_img(raster):
    image = Image.fromarray(tile_raster_images(
        X=raster,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(2, 2)))
    image.show()



def get_digit_set(digit):
    f = gzip.open('../book_code/data/mnist.pkl.gz', 'rb')
    train_set, test_set, valid_set = cPickle.load(f)
    x,y = train_set
    if digit =='all':
        return x
    x= x[y==digit]
    return x


def get_unmixed_components(digit, n_comp=10):
    ica_observations = get_digit_set(digit)
    ica_observations = ica_observations[:-1]
    # Compute ICA
    ica_observations = ica_observations - np.mean(ica_observations,axis=0)
    ica = FastICA(n_components=n_comp)
    S = ica.fit_transform(ica_observations.T)  # Reconstruct signals
    unmixing_mat = ica.components_
    A = ica.mixing_

    x = np.dot(np.dot(np.linalg.inv(np.dot(S.T,S)),S.T),ica_observations[0].T)
    #x1 = np.linalg.lstsq(S,ica_observations[0].T)

    #reconstucted_train_set = np.dot(S, A.T) + ica.mean_
    #show_img(S.T)
    return S.T



def conv_ica(digit):

    rng = np.random.RandomState(123)
    x = get_digit_set(digit)[0]
    show_img(np.reshape(x,[1,28,28]))
    n_comp = 10
    #x = np.reshape(x,[10,1,28,28])
    x = theano.shared(
            np.reshape(x,[1,1,28,28]), borrow=True
    )

    s = get_unmixed_components(digit)
    show_img(s)
    s = np.reshape(s, [n_comp,1,28,28])
    X = T.tensor4()
    W = theano.shared(
        np.asarray(
                s, dtype=theano.model_config.floatX
        ),
        borrow =True
    )
    #con = conv.conv2d(X,W,border_mode='valid',filter_flip='corr' )
    con = conv.conv2d(X,W,border_mode='valid')
    #
    # print W.get_value().shape

    f = theano.function([],con, givens={W:W,X:x})
    c = f()

    print np.shape(c)
    print c

def get_extreme(s):

    for i,m in enumerate(s):
        d = np.ma.array(m)
        d_mean = np.mean(d)
        d_std = np.std(d)
        d =np.ma.masked_inside(d, d_mean - d_std, d_mean + d_std)
        d = d.filled(0)
        s[i] = d
    return s

def get_features(feature_per_dig):
    s=[]
    for i in xrange(10):
        x = get_unmixed_components(i,feature_per_dig)
        s.append( x)
    s= np.reshape(s,[10*feature_per_dig,-1])
    return s



def conv_ica1(digit):

    x = get_digit_set(digit)[0]
    show_img(np.reshape(x,[1,28,28]))
    n_comp = 15
    x = theano.shared(
            np.reshape(x,[1,28,28]), borrow=True
    )

    s = get_unmixed_components(digit,n_comp)
    s = get_extreme(s)
    print np.shape(s)

    show_img(s)
    s = np.reshape(s, [n_comp,28,28])
    X = T.tensor3()
    W = theano.shared(
        np.asarray(
                s, dtype=theano.model_config.floatX
        ),
        borrow =True
    )
    con = conv.conv2d(X,W,border_mode='valid')
    #
    # print W.get_value().shape

    f = theano.function([],con, givens={W:W,X:x})
    c = f()

    print np.shape(c)
    print c


def conv_ica2():

    x = get_digit_set('all')
    print 'all digit set shape', np.shape(x)
    n_comp = 15
    x = theano.shared(
            np.reshape(x,[-1,28,28]), borrow=True
    )

    s = get_features(n_comp)
    s = get_extreme(s)
    #print np.shape(s)

    show_img(s)
    s = np.reshape(s, [-1,28,28])
    X = T.tensor3()
    W = theano.shared(
        np.asarray(
                s, dtype=theano.model_config.floatX
        ),
        borrow =True
    )
    con = conv.conv2d(X,W,border_mode='valid')
    #
    # print W.get_value().shape

    f = theano.function([],con, givens={W:W,X:x})
    c = f()
    c = np.reshape(c,[c.shape[0],c.shape[1]])

    print np.shape(c)





#get_unmixed_components(4)

#conv_ica1(9)
# s =np.zeros([3,2])
# s1 = np.ones([2,2])
# s= np.vstack([s,s1])
# print s

conv_ica2()