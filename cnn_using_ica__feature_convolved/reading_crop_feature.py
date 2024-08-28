import scipy.io as sio
from PIL import Image

import numpy as np

from utils import tile_raster_images


s = sio.loadmat("croped_feature.mat")
s =s['features']



#sio.savemat('mydata.mat', mdict={'whatever_data': s})



def show_img(raster, img_shape=(28,28),tile_shape = (9, 16)):
    image = Image.fromarray(tile_raster_images(
        X=raster,
        img_shape=img_shape, tile_shape=tile_shape,
        tile_spacing=(2, 2)))
    image.show()


print np.shape(s)
show_img(s,(15,15),(10,20))



