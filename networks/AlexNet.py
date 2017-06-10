from kaffe.tensorflow import Network
import numpy as np
import skimage.transform

class AlexNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5'))

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
    def preprocess_image(im, keep_resolution=False):
        # assumes RGB
        if !keep_resolution:
            im = np.copy(im).astype('uint8')
            # Resize so smallest dim = 256, preserving aspect ratio
            h, w, _ = im.shape
            if h < w:
                im = skimage.transform.resize(im, (256, int(w*256/h)), preserve_range=True)
            else:
                im = skimage.transform.resize(im, (int(h*256/w), 256), preserve_range=True)

            # Central crop to 224x224
            h, w, _ = im.shape
            im = im[h//2-113:h//2+114, w//2-113:w//2+114]


        # Convert to BGR
        im = im[:, :, ::-1]

        # substract mean
        im = im - AlexNet.mean()

        return im
