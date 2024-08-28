



import gzip

import numpy as np
from PIL import Image

import cPickle

import theano
from sklearn.decomposition import FastICA
from theano.tensor.signal import downsample

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
    ica_observations = ica_observations[:-1]
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
    for i in xrange(10):
        x = get_unmixed_components(i,feature_per_dig)
        s.append( x)
    s= np.reshape(s,[10*feature_per_dig,-1])
    s = get_extreme(s)
    #print np.shape(s)

    show_img(s)

    s = np.reshape(s, [-1,28,28])
    s = [np.rot90(r,2) for r in s]
    s = [r[2:-2,2:-2] for r in s]
    return s


def get_conv_feature(s, x):

    x = theano.shared(
            np.reshape(x,[-1,1,28,28]), borrow=True
    )
    X = T.tensor4()
    s = np.reshape(s, [-1,1,24,24])
    W = theano.shared(
        np.asarray(
                s, dtype=theano.model_config.floatX
        ),
        borrow =True
    )

    #conv_out = theano.tensor.nnet.conv2d(X,W,border_mode='valid', filter_flip=False)
    conv_out = conv_nnet.conv2d(X,W,border_mode='valid')
    pooled_out = downsample.max_pool_2d(
        input=conv_out,
        ds=[5,5],
        ignore_border=True
    )


    c1 =  theano.tensor.tanh(pooled_out)
    c2 =  theano.function([], c1, givens={W:W, X:x})
    c = c2()
    print np.shape(c)
    c = np.reshape(c,[c.shape[0],c.shape[1]])


    return c


def get_conv_feature1(s, x):

    x = theano.shared(
            np.reshape(x,[-1,1,28,28]), borrow=True
    )

    X = T.tensor4()
    s = np.reshape(s, [-1,1,28,28])
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


def load_prepared_dataset(n_comp):

    print '*********************************************************************************************************88'
    s = get_basis_features(n_comp)

    train_set_x, train_set_y = get_digit_set('all','train')
    train_set_feature = get_conv_feature(s, train_set_x)
    print 'train set feature shape', np.shape(train_set_feature)


    test_set_x , test_set_y = get_digit_set('all','test')
    test_set_feature = get_conv_feature(s, test_set_x)
    print 'test set feature shape', np.shape(test_set_feature)


    valid_set_x, valid_set_y = get_digit_set('all','valid')
    valid_set_feature = get_conv_feature(s, valid_set_x)
    print 'valid set feature shape', np.shape(valid_set_feature)


    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.model_config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.model_config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset((train_set_feature,train_set_y))
    test_set_x, test_set_y = shared_dataset((test_set_feature,test_set_y))
    valid_set_x, valid_set_y = shared_dataset((valid_set_feature,valid_set_y))


    return [(train_set_x, train_set_y),(test_set_x, test_set_y),(valid_set_x, valid_set_y)]





