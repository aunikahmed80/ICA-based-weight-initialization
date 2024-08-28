import cv2
import numpy as np
from scipy.signal import convolve2d

class ElasticDistortion:
    def __init__(self, kernel_dim, sd):
        self.kernel = self.makeKernel(kernel_dim, sd)
        # check if kernel dimesnion is odd
        if kernel_dim % 2 == 0:
            raise ValueError("Kernel dimension should be odd")



    def makeKernel(self, dim, sd):
        # make a dim x dim continuous Gaussian kernel
        if (dim % 2 == 0):
            Exception("kernel dim must be odd")


        center = dim / 2; # note truncation
        coef = 1.0 / (2.0 * np.pi * (sd * sd)); # normalizes so all value sum to (close) 1.0
        denom = 2.0 * (sd * sd);
        sum = 0.0; # for more accurate normalization
        result = np.zeros(shape  = [dim,dim])
        for i in xrange(dim):
            for j in xrange(dim):
                x = np.abs(center - j);
                y = np.abs(center - i);
                num = -1.0 * ((x * x) + (y * y));
                z = coef * np.exp(num / denom);
                result[i,j] = z;
                sum += z

        result /= sum

        return result;


    def MakeDisplace(self,width, height, kernel, intensity):
        dField = np.random.uniform(-1,1,(height,width))
        dField = intensity * convolve2d(dField, kernel) # Smooth
        return dField


    def Displace(self,image, xField, yField):
        # take each pixel value and replace it with a val detemined by the x- and y- fields

        resultPixels = result = np.zeros(image.shape)

        #int picker = 0;
        for row in xrange(image.shape[1]):
            for col in xrange(image.shape[0]):     #int ii = i + (int)yField[i][j];
                #int jj = j + (int)xField[i][j];

                lowii = row + (int)(np.floor(yField[row,col]))
                hiii = row + (int)(np.ceil(yField[row,col]))
                lowjj = col + (int)(np.floor(xField[row,col]))
                hijj = col + (int)(np.ceil(xField[row,col]))

                if  lowii < 0 or lowjj < 0 or hiii > image.shape[1] -1 or hijj > image.shape[0] - 1:
                    result[row, col] = 0
                    continue
                else :
                    sum = image[lowii][lowjj] + image[lowii][hijj] + image[hiii][lowjj] + image[hiii][hijj]
                    avg = np.round(sum / 4.0);
                    resultPixels[row, col] = avg #// avg value produces a smoothing effect
        return resultPixels;




    def elastic_transform(self,image, alpha=36, negated=False):

        # convert the image to single channel if it is multi channel one
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check if the image is a negated one
        if not negated:
            image = 255-image

        # check if the image is a square one
        if image.shape[0] != image.shape[1]:
            raise ValueError("Image should be of sqaure form")


        kernel = self.kernel
        displace_fieldX = self.MakeDisplace(width = image.shape[0], height = image.shape[1], kernel = kernel, intensity = alpha)
        displace_fieldY = self.MakeDisplace(width = image.shape[0], height = image.shape[1], kernel = kernel, intensity = alpha)
        result = self.Displace(image = image, xField= displace_fieldX, yField= displace_fieldY)
        # create an em
        # if the input image was not negated, make the output image also a non
        # negated one
        if not negated:
            result = 255-result
        #save_vector_field(displacement_field_x, displacement_field_y)
        return result,[displace_fieldX,displace_fieldY]



















#print makeKernel(5,3)
