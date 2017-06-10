from kaffe.tensorflow import Network
import math
from PIL import Image
import numpy as np

class Flownet_Simple(Network):
    def setup(self):
        (self.feed('data1', 
                   'data2')
             .concat(3, name='Concat1')
             .conv(7, 7, 64, 2, 2, name='conv1')
             .conv(5, 5, 128, 2, 2, name='conv2')
             .conv(5, 5, 256, 2, 2, name='conv3')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 512, 2, 2, name='conv4')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 2, 2, name='conv5')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 1024, 2, 2, name='conv6')
             .conv(3, 3, 1024, 1, 1, name='conv6_1')
             .conv(3, 3, 2, 1, 1, relu=False, name='Convolution1')
             .deconv(4, 4, 2, 2, 2,  relu=False, name='upsample_flow6to5'))

        (self.feed('conv6_1')
             .deconv(4, 4, 512, 2, 2,  name='deconv5'))

        (self.feed('upsample_flow6to5', 
                   'conv5_1', 
                   'deconv5')
             .concat(3, name='Concat2')
             .conv(3, 3, 2, 1, 1, relu=False, name='Convolution2')
             .deconv(4, 4, 2, 2, 2,  relu=False, name='upsample_flow5to4'))

        (self.feed('Concat2')
             .deconv(4, 4, 256, 2, 2,  name='deconv4'))

        (self.feed('upsample_flow5to4', 
                   'conv4_1', 
                   'deconv4')
             .concat(3, name='Concat3')
             .conv(3, 3, 2, 1, 1, relu=False, name='Convolution3')
             .deconv(4, 4, 2, 2, 2, relu=False, name='upsample_flow4to3'))

        (self.feed('Concat3')
             .deconv(4, 4, 128, 2, 2,  name='deconv3'))

        (self.feed('upsample_flow4to3', 
                   'conv3_1', 
                   'deconv3')
             .concat(3, name='Concat4')
             .conv(3, 3, 2, 1, 1, relu=False, name='Convolution4')
             .deconv(4, 4, 2, 2, 2, relu=False, name='upsample_flow3to2'))

        (self.feed('Concat4')
             .deconv(4, 4, 64, 2, 2, name='deconv2'))

        (self.feed('upsample_flow3to2', 
                   'conv2', 
                   'deconv2')
             .concat(3, name='Concat5')
             .conv(3, 3, 2, 1, 1, relu=False, name='Convolution5')
             .add(coeff=20.0, name='Eltwise4')
             .conv(1, 1, 2, 1, 1, relu=False, name='Convolution6'))

    @staticmethod
    def mean():
        FLOWNET_SCALE = 0.0039216
        FLOWNET_MEAN1 = [ 0.411451 * FLOWNET_SCALE, 0.432060 * FLOWNET_SCALE,  0.450141 * FLOWNET_SCALE ]
        FLOWNET_MEAN2 = [ 0.410602 * FLOWNET_SCALE, 0.431021 * FLOWNET_SCALE,  0.448553 * FLOWNET_SCALE ]
        return np.array(FLOWNET_MEAN1)

    @staticmethod
    def make_prod64(n):
        n = int(n)
        h = math.floor(n / 64) * 64
        if n % 64 != 0:
            h += 64

        return h

    @staticmethod
    def preprocess_image(im, keep_resolution):
        # unfortunately, we can't keep resolution the same in flow net

        # flownet requires image height and width to be a multiple of 64
        height = Flownet_Simple.make_prod64(im.shape[0])
        width = Flownet_Simple.make_prod64(im.shape[1])

        # resize
        im = Image.fromarray(im, 'RGB')
        im = im.resize((width, height), resample=Image.BILINEAR)
        im = np.asarray(im, dtype="float32")

        # convert to BGR
        im = im[:,:,::-1]

        # substract MEAN
        # note - there are two means in flownet, we are ignoring one of them. They are close
        im = im - Flownet_Simple.mean()

        return im

    @staticmethod
    def preprocess_shape(input_shape, keep_resolution=False):
        # unfortunately, we can't keep resolution the same in flow net
        # flownet requires image height and width to be a multiple of 64

        output_shape = input_shape
        output_shape[0] = Flownet_Simple.make_prod64(input_shape[0])
        output_shape[1] = Flownet_Simple.make_prod64(input_shape[1])
        
        return output_shape

    def single_image():
        return False
