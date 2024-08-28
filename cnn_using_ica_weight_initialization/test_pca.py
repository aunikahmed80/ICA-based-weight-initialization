import gzip

import Image
import cPickle
import numpy as np
from sklearn.decomposition import PCA, FastICA

from ica_code.utils import tile_raster_images


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

def ica_in_pca_domain():
    d,dy = get_digit_set(8)

    pca = PCA(n_components=64)
    x = pca.fit_transform(d)
    print np.shape(x)
    # Compute ICA
    meu = np.mean(x,axis=0)
    print np.shape(meu)
    ica_observations = x - meu
    ica = FastICA(n_components=20)
    S = ica.fit_transform(ica_observations.T)   # Reconstruct signals


    #S = pca.inverse_transform(S.T)
    # #show_img(d[0:10])
    print np.shape(S.T)
    #show_img(S.T[0:10])


    return S.T

ica_in_pca_domain()