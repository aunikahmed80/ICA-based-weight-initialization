import gzip

import cPickle

from utils import tile_raster_images
from my_mnist_helper import *
import cv2

import numpy as np
import pylab
import matplotlib.cm as cm
import Image

import scipy.io as sio



def get_croped_feature_from_file():

    s = sio.loadmat("croped_featues_1.06_.95")
    s =s['crop_features']
    return s





def get_vector_field_image(vx, vy):

    # use LaTeX, choose nice some looking fonts and tweak some settings
    matplotlib.rc('font', family='serif')
    matplotlib.rc('font', size=16)
    matplotlib.rc('legend', fontsize=16)

    matplotlib.interactive(False)
    close('all')
    f1 = plt.figure(1,figsize=(4,4))
    vx = np.flipud(vx)
    vy = np.flipud(vy)
    quiver( vx, vy, pivot='middle', headwidth=4, headlength=6)
    xlabel('$x$')
    ylabel('$y$')
    axis('image')
    #return im;
    f1.show()



def get_img(raster, img_shp=(15,15), tile_shp=(1,1)):
    image = Image.fromarray(tile_raster_images(
        X=raster,
        img_shape=img_shp, tile_shape=tile_shp,
        tile_spacing=(2, 2)))
    #image.show()
    return image



def show_modified_img(raster):

    kernel_dim =11
    feature_img = get_img(np.reshape(raster,(1,-1)))
    distorted_image,elastic_field = elastic_transform(np.reshape(raster,(15,15)), kernel_dim=kernel_dim,
                                        alpha=30,
                                        sigma=8,negated=True)
    distorted_image = get_img(np.reshape(distorted_image,(1,-1)))


    img_offset = int(kernel_dim/2)
    #get_vector_field_image(elastic_field[0][img_offset:-img_offset,img_offset:-img_offset],elastic_field[1][img_offset:-img_offset,img_offset:-img_offset])

    f2 = plt.figure()
    pylab.subplot(2,1,1)
    pylab.imshow(feature_img,cmap=cm.Greys_r)
    pylab.subplot(2,1,2)
    pylab.imshow(distorted_image,cmap=cm.Greys_r)



#
# cropped_feature = get_croped_feature_from_file()
# for i in xrange(5):
#     show_modified_img(cropped_feature[0])
#
#
# plt.show()

x = get_croped_feature_from_file()
x= np.array(x)
#numpy.set_printoptions(precision=3, threshold= 10000000)
print x.min() , ' ', x.max()


#print x[0]