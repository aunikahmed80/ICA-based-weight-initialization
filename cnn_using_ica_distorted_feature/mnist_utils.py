import Image
import gzip

import cPickle


from utils import tile_raster_images


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

def show_img(raster,img_shp=(28,28),tile_shp=(10,10)):
    image = Image.fromarray(tile_raster_images(
        X=raster,
        img_shape=img_shp, tile_shape=tile_shp,
        tile_spacing=(2, 2)))
    image.show()
