import skimage.transform
from kaffe.tensorflow import Network
import numpy as np

class vgg16(Network):
    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
             .max_pool(2, 2, 2, 2, name='pool5'))

        if self.fixed_resolution:
            (self.feed('pool5')
                .fc(4096, name='fc6')
                .fc(4096, name='fc7')
                .fc(1000, relu=False, name='fc8')
                .softmax(name='prob'))

    def single_image():
        return True

    @staticmethod
    def mean():
        # Pixel mean values (BGR order) as a (1, 1, 3) array
        # These are the values originally used for training VGG16
        return np.array([[103.939, 116.779, 123.68]])

    @staticmethod
    def preprocess_image(im, fixed_resolution=True):
        if fixed_resolution:
            im = np.copy(im).astype('uint8')
            # Resize so smallest dim = 256, preserving aspect ratio
            h, w, _ = im.shape
            if h < w:
                im = skimage.transform.resize(im, (256, int(w*256/h)), preserve_range=True)
            else:
                im = skimage.transform.resize(im, (int(h*256/w), 256), preserve_range=True)

            # Central crop to 224x224
            h, w, _ = im.shape
            im = im[h//2-112:h//2+112, w//2-112:w//2+112]



        # Convert to BGR
        im = im[:, :, ::-1]

        # substract mean
        im = im - vgg16.mean()

        return im
