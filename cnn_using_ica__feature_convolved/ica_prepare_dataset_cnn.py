



import gzip
import scipy.io as sio
import numpy as np
from PIL import Image

import cPickle

import theano
from sklearn.decomposition import FastICA
from theano.tensor.signal import downsample

from utils import tile_raster_images

from theano import tensor as T

from theano.tensor.nnet import conv as conv_nnet


def show_img(raster, img_shape=(28,28),tile_shape = (9, 16)):
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
    x= x[y == digit]
    y = y[y == digit]


    return (x,y)



def get_conv_feature(s, x):

    x =  theano.shared(
            np.reshape(x,[-1,1,28,28]), borrow=True
    )
    X = T.tensor4()
    basis_feature_shape = np.shape(s)
    s = np.reshape(s, [-1,1,basis_feature_shape[1],basis_feature_shape[2]])
    W =  theano.shared(
        np.asarray(
                s, dtype= theano.model_config.floatX
        ),
        borrow =True
    )

    conv_out = theano.tensor.nnet.conv2d(X,W,border_mode='valid', filter_flip= False)
    #conv_out = conv_nnet.conv2d(X,W,border_mode='valid')
    pooled_out = downsample.max_pool_2d(
        input=conv_out,
        ds=[2,2],
        ignore_border=True
    )


    conv_out_f =  theano.function([], pooled_out, givens={W:W, X:x})
    c = conv_out_f()

    return c


def load_ica_conv_dataset_after_pooling():

    s =get_crop_basis_feature_from_file()
    #s = get_basis_features_from_file()

    train_set_feature ,train_set_y= get_train_conv_feature(s)
    print 'train set feature shape', np.shape(train_set_feature)

    test_set_x , test_set_y = get_digit_set('all','test')
    test_set_feature = get_conv_feature(s, test_set_x)

    print 'test set feature shape', np.shape(test_set_feature)

    valid_set_x, valid_set_y = get_digit_set('all','valid')
    valid_set_feature = get_conv_feature(s, valid_set_x)
    print 'valid set feature shape', np.shape(valid_set_feature)

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x =  theano.shared(np.asarray(data_x,
                                             dtype= theano.model_config.floatX),
                                  borrow=borrow)
        shared_y =  theano.shared(np.asarray(data_y,
                                             dtype= theano.model_config.floatX),
                                  borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset((train_set_feature,train_set_y))
    test_set_x, test_set_y = shared_dataset((test_set_feature,test_set_y))
    valid_set_x, valid_set_y = shared_dataset((valid_set_feature,valid_set_y))

    return [(train_set_x, train_set_y),(test_set_x, test_set_y),(valid_set_x, valid_set_y)]


def save_basis_feature(s):
    f = open('ica_basis_feature_temp','wb')
    cPickle.dump(s,f , -1)
    f.close()


def get_basis_features_from_file(filename = 'ica_basis_feature_20_1.0_0.93'):
    f = open(filename)
    s = cPickle.load(f)
    f.close()
    return


def get_crop_basis_feature_from_file():

    s = sio.loadmat("croped_featues_1.06_.95")
    s =s['crop_features']

    s = np.reshape(s, [-1,15,15])
    #s = [np.rot90(r,2) for r in s]
    return  s


def get_train_conv_feature(s):
    train_conv_set_x = []

    x, y = get_digit_set('all','train')
    for i in xrange(5):
        start_index = i*10000
        end_index = (i+1)*10000
        train_set_feature = get_conv_feature(s, x[start_index:end_index])
        print np.shape(train_set_feature)
        train_conv_set_x= np.append(np.reshape(train_conv_set_x,[-1,200,7,7]),train_set_feature,axis=0)

    print np.shape(train_conv_set_x)
    return [train_conv_set_x,y]


# s =get_crop_basis_feature_from_file()
# show_img(s,(15,15),(10,20))
# train_set_x, train_set_y = get_train_conv_feature(s)
#load_ica_conv_dataset_after_pooling()