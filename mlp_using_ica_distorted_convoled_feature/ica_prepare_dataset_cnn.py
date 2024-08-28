



import gzip

import gc
import numpy as np
from PIL import Image

import cPickle

import theano
from sklearn import preprocessing
from sklearn.decomposition import FastICA
from theano.tensor.signal import downsample

import my_mnist_helper
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
    # Compute ICA
    ica_observations = ica_observations - np.mean(ica_observations,axis=0)
    ica = FastICA(n_components=n_comp, max_iter=400)
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
    print np.shape(c)
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

    #s = get_basis_features(n_comp)


    s = get_non_rotated_basis_features_from_file()
    #save_basis_feature(s, n_comp)
    #save_basis_feature_matlab_format('basis_feature', 'features',s)

    train_set_x, train_set_y = get_digit_set('all','train')
    train_set_feature = get_conv_feature(s, train_set_x)

    #train_set_x, train_set_y = get_distorted_mnist_from_file()
    #train_set_feature = get_conv_feature(s, train_set_x[0:50000])
    #train_set_feature =np.append(train_set_feature,get_conv_feature(s, train_set_x[50000:100000]),0)

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
    #print test_set_y.eval()[0:10]
    return [(train_set_x, train_set_y),(test_set_x, test_set_y),(valid_set_x, valid_set_y)]


def save_basis_feature(s, n_comp):
    f = open('ica_basis_feature_temp','wb')
    cPickle.dump(s,f , -1)
    f.close()



def save_basis_feature_matlab_format(file_name, dataset_name,data):
    sio.savemat(file_name, mdict={dataset_name: data})


def get_non_rotated_basis_features_from_file(filename = 'ica_basis_feature_0.91_0.93'):
    f = open(filename)
    s = cPickle.load(f)
    f.close()
    return s



def save_distorted_mnist_dataset(dataset):
    f = open('mnist_distorted', 'wb')
    cPickle.dump(dataset, f, -1)
    f.close()

def get_distorted_mnist_from_file(filename = 'mnist_distorted'):
    f = open(filename)
    mnist_distorted = cPickle.load(f)
    f.close()
    return mnist_distorted



if __name__ == '__main__':
    mnist_distorted = get_distorted_mnist_from_file()
    train = mnist_distorted[0]
    show_img(train[0:200])
#
# s = get_non_rotated_basis_features_from_file()
# print np.min(s), np.max(s)
# min = np.min(s)
# max = np.max(s)
# s = -1 + 2*(s - min)/(max-min) #scaling between (-1,1)
# print np.min(s), np.max(s)