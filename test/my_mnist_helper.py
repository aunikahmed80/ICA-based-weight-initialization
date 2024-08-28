import gc
import matplotlib
import numpy
import math

from numpy.random import random_integers
from scipy.signal import convolve2d
import matplotlib
from pylab import *
#from theano import tensor
#import theano



def create_2d_gaussian(dim, sigma):
    """
    This function creates a 2d gaussian kernel with the standard deviation
    denoted by sigma

    :param dim: integer denoting a side (1-d) of gaussian kernel
    :type dim: int

    :param sigma: the standard deviation of the gaussian kernel
    :type sigma: float

    :returns: a numpy 2d array
    """

    # check if the dimension is odd
    if dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    # initialize the kernel
    kernel = numpy.zeros((dim, dim), dtype=numpy.float16)

    # calculate the center point
    center = dim/2

    # calculate the variance
    variance = sigma ** 2

    # calculate the normalization coefficeint
    coeff = 1. / (2 * variance)

    # create the kernel
    for x in range(0, dim):
        for y in range(0, dim):
            x_val = abs(x - center)
            y_val = abs(y - center)
            numerator = x_val**2 + y_val**2
            denom = 2*variance

            kernel[x,y] = coeff * numpy.exp(-1. * numerator/denom)

    # normalise it
    return kernel/sum(sum(kernel))

def get_conv_feature(s, x):

    digits = theano.shared(
            np.reshape(x,[-1,1,28,28]), borrow=True
    )
    X = tensor.tensor4()
    kernel_shape = np.shape(s)
    s = np.reshape(s, [-1,1,kernel_shape[0],kernel_shape[1]])
    W = theano.shared(
        np.asarray(
                s, dtype=theano.model_config.floatX
        ),
        borrow =True
    )
    conv_out = tensor.nnet.conv2d(X,W,border_mode='full')


    c2 = theano.function([], conv_out, givens={W:W,X:digits})
    c = c2()

    gc.collect()

    return c



def elastic_transform(image,  kernel, alpha=36, negated=False ):
    """
    This method performs elastic transformations on an image by convolving
    with a gaussian kernel.

    NOTE: Image dimensions should be a sqaure image

    :param image: the input image
    :type image: a numpy nd array

    :param kernel_dim: dimension(1-D) of the gaussian kernel
    :type kernel_dim: int

    :param sigma: standard deviation of the kernel
    :type sigma: float

    :param alpha: a multiplicative factor for image after convolution
    :type alpha: float

    :param negated: a flag indicating whether the image is negated or not
    :type negated: boolean

    :returns: a nd array transformed image
    """

    # convert the image to single channel if it is multi channel one
    # check if the image is a negated one
    if not negated:
        image = 255-image

    # check if the image is a square one
    if image.shape[0] != image.shape[1]:
        raise ValueError("Image should be of sqaure form")


    # create an empty image
    result = numpy.zeros(image.shape,dtype='float32')

    # create random displacement fields

    displacement_field_x = alpha * numpy.random.uniform(-1,1,(image.shape[0],image.shape[0]))
    displacement_field_y = alpha * numpy.random.uniform(-1,1,(image.shape[0],image.shape[0]))

    # create the gaussian kernel


    # convolve the fields with the gaussian kernel
    displacement_field_x = convolve2d(displacement_field_x, kernel)
    displacement_field_y = convolve2d(displacement_field_y, kernel)

    # displacement_field_x = get_conv_feature(kernel,displacement_field_x)
    # displacement_field_y = get_conv_feature(kernel,displacement_field_y)


    # make the distortrd image by averaging each pixel value to the neighbouring
    # four pixels based on displacement fields

    for row in range(image.shape[1]):
        for col in range(image.shape[0]):
            low_ii = numpy.int(numpy.floor(row + displacement_field_x[row, col]))
            high_ii = numpy.int(numpy.floor(row + displacement_field_x[row, col]))

            low_jj = numpy.int(numpy.floor(col + displacement_field_y[row, col]))
            high_jj = numpy.int(numpy.floor(col + displacement_field_y[row, col]))

            if low_ii < 0 or low_jj < 0 or high_ii >= image.shape[1] -1 \
               or high_jj >= image.shape[0] - 1:
                result[row, col] = 0
                continue

            x1 = image[low_ii, low_jj] + (displacement_field_x[row, col] % 1) *(image[low_ii, high_jj] - image[low_ii, low_jj] )
            x2 = image[high_ii, low_jj] + (displacement_field_x[row, col]%1) *(image[high_ii, high_jj]-image[high_ii, low_jj])
            res = x1+ (displacement_field_y[row, col] %1 )*(x2-x1)
            result[row, col] = res

    # if the input image was not negated, make the output image also a non
    # negated one
    if not negated:
        result = 255-result
    #save_vector_field(displacement_field_x, displacement_field_y)
    return result,[displacement_field_x,displacement_field_y]

