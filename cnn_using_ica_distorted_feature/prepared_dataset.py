
import gzip
import timeit

import gc
import numpy as np
from PIL import Image

import theano
from theano.tensor.signal import downsample

import my_mnist_helper

from utils import tile_raster_images
import scipy.io as sio
from theano import tensor as T
import mnist_utils


def get_croped_feature_from_file():

    s = sio.loadmat("croped_featues_1.06_.95")
    s =s['crop_features']
    return s


def get_all_distored_features(cropped_features, n_distortion ):


    distored_feature_dim = 15
    kernel_dim = 11
    sigma = 8
    kernel = my_mnist_helper.create_2d_gaussian(kernel_dim, sigma)
    distored_features = np.zeros([len(cropped_features)*n_distortion,distored_feature_dim,distored_feature_dim])
    n =0
    for feature_num in xrange(len(cropped_features)):
        for distion_num in xrange(n_distortion):
            raster = cropped_features[feature_num]
            distorted_feature,elastic_field = my_mnist_helper.elastic_transform(np.reshape(raster, (15, 15)),
                                                                                kernel,
                                                                                alpha=20,
                                                                                 negated=True)
            distored_features[n] = distorted_feature
            n+=1
    return distored_features



def get_conv_feature(s, x, pool_dim):

    digits = theano.shared(
            np.reshape(x,[-1,1,28,28]), borrow=True
    )
    X = T.tensor4()
    basis_feature_shape = np.shape(s)
    s = np.reshape(s, [-1,1,basis_feature_shape[1],basis_feature_shape[2]])
    W = theano.shared(
        np.asarray(
                s, dtype=theano.model_config.floatX
        ),
        borrow =True
    )
    conv_out = T.nnet.conv2d(X,W,border_mode='valid', filter_flip= False)
    pooled_out = downsample.max_pool_2d(
        input=conv_out,
        ds=[pool_dim,pool_dim],
        ignore_border=True
    )


    c2 = theano.function([], pooled_out, givens={W:W,X:digits})
    c = c2()

    gc.collect()


    return c



def get_max_feature(distored_features, x,n_distortion , pool_dim):
    n_cropped_feature =200
    kernel_dim = 13
    sigma = 8

    distored_features_dim = np.shape(distored_features)[1]
    pooled_feature_dim = (28 - distored_features_dim +1)/pool_dim
    convolved_distort_features = np.zeros([ len(distored_features)/n_distortion, len(x), pooled_feature_dim, pooled_feature_dim])

    for i in xrange(n_cropped_feature):
        print 'batch: ', i+1
        distored_feature_batch = distored_features[i*n_distortion:(i+1)*n_distortion]

        convolved_distort_feature_batch = get_conv_feature(distored_feature_batch, x, pool_dim )
        max_feaature = convolved_distort_feature_batch.max(1)
        convolved_distort_features[i] = max_feaature

    convolved_distort_features = np.swapaxes(convolved_distort_features,0,1)
    return convolved_distort_features


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



def load_ica_distort_conv_dataset_after_pooling():
    n_distortion =10
    pool_dim =2

    cropped_features = get_croped_feature_from_file()
    distored_features = get_all_distored_features(cropped_features, n_distortion)

    train_set_x, train_set_y = mnist_utils.get_digit_set('all','train')
    train_set_x, train_set_y = digit_set_with_distortion(train_set_x,train_set_y)
    train_convolved_distort_feature = get_max_feature(distored_features, train_set_x, n_distortion ,pool_dim)
    print 'train set feature shape', np.shape(train_convolved_distort_feature)

    test_set_x, test_set_y = mnist_utils.get_digit_set('all','test')
    test_convolved_distort_feature = get_max_feature(distored_features, test_set_x, n_distortion ,pool_dim)
    print 'test set feature shape', np.shape(test_convolved_distort_feature)

    valid_set_x, valid_set_y = mnist_utils.get_digit_set('all','valid')
    valid_convolved_distort_feature = get_max_feature(distored_features, valid_set_x, n_distortion ,pool_dim)
    print 'valid set feature shape', np.shape(valid_convolved_distort_feature)

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x =  theano.shared(np.asarray(data_x,
                                             dtype= theano.model_config.floatX),
                                  borrow=borrow)
        shared_y =  theano.shared(np.asarray(data_y,
                                             dtype= theano.model_config.floatX),
                                  borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset((train_convolved_distort_feature,train_set_y))
    test_set_x, test_set_y = shared_dataset((test_convolved_distort_feature,test_set_y))
    valid_set_x, valid_set_y = shared_dataset((valid_convolved_distort_feature,valid_set_y))

    return [(train_set_x, train_set_y),(test_set_x, test_set_y),(valid_set_x, valid_set_y)]

# start_time = timeit.default_timer()
# load_ica_conv_dataset_after_pooling()
# end_time = timeit.default_timer()
# print 'convolution takes ', (end_time - start_time)/60.
#
# test_set_x, test_set_y = mnist_utils.get_digit_set('all','test')
# digit_set_with_distortion(test_set_x,test_set_y)

#load_ica_distort_conv_dataset_after_pooling()