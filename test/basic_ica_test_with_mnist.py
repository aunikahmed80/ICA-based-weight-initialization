import gzip
import numpy as np
from PIL import Image
import pickle as cPickle
from sklearn.decomposition import FastICA

from utils import tile_raster_images

def get_digit_set():
    f = gzip.open('../../data/mnist.pkl.gz', 'rb')
    train_set, test_set, valid_set = cPickle.load(f,encoding='latin1')
    x,y = train_set

    x1 = x[y==5 ]
    x2 = x[y==3]
    x = np.append(x1,x2,axis=0)
    return (x,y)

def show_img(raster):
    image = Image.fromarray(tile_raster_images(
        X=raster,
        img_shape=(28, 28), tile_shape=(2, 2),
        tile_spacing=(2, 2)))
    image.show()



def show_reconstraction(n_comp=10):
    ica_observations, ica_observations_y = get_digit_set()
    # Compute ICA
    ica = FastICA(n_components=n_comp)
    S = ica.fit_transform(ica_observations.T)  # Reconstruct signals
    x = np.dot(S, ica.mixing_.T) #+ ica.mean_
    print(S[0, 0])
    S[0,0] = 0
    x1 = np.dot(S, ica.mixing_.T)

    show_img(x[0:2])
    return S.T

show_reconstraction(2)