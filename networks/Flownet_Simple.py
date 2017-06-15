from kaffe.tensorflow import Network
import math
from PIL import Image
import numpy as np

class Flownet_Simple(Network):
    def setup(self):
        input_shape = self.layers['data1'].get_shape().as_list()
        height = input_shape[1]
        width = input_shape[2]
        target_height = self.make_prod64(height)
        target_width = self.make_prod64(width)
        height_coeff = float(height) / float(target_height)
        width_coeff = float(width) / float(target_width)

        (self.feed('data1')
            .subtract_mean(input_scale=0.0039216,
                means=[0.411451, 0.432060, 0.450141], name='img0_aug')
            .resize(target_height, target_width, name='Resample1'))

        (self.feed('data2')
            .subtract_mean(input_scale=0.0039216,
                means=[0.410602, 0.431021, 0.448553], name='img1_aug')
            .resize(target_height, target_width, name='Resample2'))

        (self.feed('Resample1', 'Resample2')
             .concat(3, name='Concat1'))

        (self.feed('Concat1')
             .conv(7, 7, 64, 2, 2, name='conv1', alpha=0.1))

        return

        (self.feed('conv1')
             .conv(5, 5, 128, 2, 2, name='conv2', alpha=0.1)
             .conv(5, 5, 256, 2, 2, name='conv3', alpha=0.1)
             .conv(3, 3, 256, 1, 1, name='conv3_1', alpha=0.1)
             .conv(3, 3, 512, 2, 2, name='conv4', alpha=0.1)
             .conv(3, 3, 512, 1, 1, name='conv4_1', alpha=0.1)
             .conv(3, 3, 512, 2, 2, name='conv5', alpha=0.1)
             .conv(3, 3, 512, 1, 1, name='conv5_1', alpha=0.1)
             .conv(3, 3, 1024, 2, 2, name='conv6', alpha=0.1)
             .conv(3, 3, 1024, 1, 1, name='conv6_1', alpha=0.1)
             .conv(3, 3, 2, 1, 1, relu=False, name='Convolution1')
             .deconv(4, 4, 2, 2, 2,  relu=False, name='upsample_flow6to5'))

        (self.feed('conv6_1')
             .deconv(4, 4, 512, 2, 2,  name='deconv5', alpha=0.1))

        (self.feed('upsample_flow6to5', 
                   'conv5_1', 
                   'deconv5')
             .concat(3, name='Concat2')
             .conv(3, 3, 2, 1, 1, relu=False, name='Convolution2')
             .deconv(4, 4, 2, 2, 2,  relu=False, name='upsample_flow5to4'))

        (self.feed('Concat2')
             .deconv(4, 4, 256, 2, 2,  name='deconv4', alpha=0.1))

        (self.feed('upsample_flow5to4', 
                   'conv4_1', 
                   'deconv4')
             .concat(3, name='Concat3')
             .conv(3, 3, 2, 1, 1, relu=False, name='Convolution3')
             .deconv(4, 4, 2, 2, 2, relu=False, name='upsample_flow4to3'))

        (self.feed('Concat3')
             .deconv(4, 4, 128, 2, 2,  name='deconv3', alpha=0.1))

        (self.feed('upsample_flow4to3', 
                   'conv3_1', 
                   'deconv3')
             .concat(3, name='Concat4')
             .conv(3, 3, 2, 1, 1, relu=False, name='Convolution4')
             .deconv(4, 4, 2, 2, 2, relu=False, name='upsample_flow3to2'))

        (self.feed('Concat4')
             .deconv(4, 4, 64, 2, 2, name='deconv2', alpha=0.1))

        (self.feed('upsample_flow3to2', 
                   'conv2', 
                   'deconv2')
             .concat(3, name='Concat5')
            .conv(3, 3, 2, 1, 1, relu=False, name='Convolution5')
            .add(coeff=20.0, name='Eltwise4')
            .resize(height, width, name='Resample4')
            .conv(1, 1, 2, 1, 1, relu=False, biased=False, 
                    weights = [[height_coeff, 0], [0, width_coeff]],
                    name='Convolution6'))

    def make_prod64(self, n):
        n = int(n)
        h = math.floor(n / 64) * 64
        if n % 64 != 0:
            h += 64

        return h

    @staticmethod
    def preprocess_image(im, keep_resolution):
        # convert to BGR
        im = im[:,:,::-1]

        return im

    @staticmethod
    def postprocess_image(im, width, height):
        return im

    @staticmethod
    def single_image():
        return False
