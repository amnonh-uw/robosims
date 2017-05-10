from kaffe.tensorflow import Network

class FCN_RGB(Network):
    def setup(self):
        (self.feed('image_data')
             .conv(9, 9, 64, 1, 1, relu=False, name='conv0')
             .max_pool(3, 3, 2, 2, name='pool0')
             #.batch_normalization(name='bn0')
             .prelu(name='relu0')
             .conv(7, 7, 64, 1, 1, relu=False, name='conv1')
             .max_pool(3, 3, 2, 2, name='pool1')
             #.batch_normalization( name='bn1')
             .prelu(name='relu1')
             .conv(5, 5, 64, 1, 1, relu=False, name='conv2')
             .max_pool(3, 3, 2, 2, name='pool2')
             #.batch_normalization( name='bn2')
             .prelu(name='relu2'))

        (self.feed('image_data_pert')
             .conv(9, 9, 64, 1, 1, relu=False, name='conv_per0')
             .max_pool(3, 3, 2, 2, name='pool_per0')
             #.batch_normalization(name='bn_per0')
             .prelu(name='relu_per')
             .conv(7, 7, 64, 1, 1, relu=False, name='conv_per1')
             .max_pool(3, 3, 2, 2, name='pool_per1')
             #.batch_normalization( name='bn_per1')
             .prelu(name='relu_per1')
             .conv(5, 5, 64, 1, 1, relu=False, name='conv_per2')
             .max_pool(3, 3, 2, 2, name='pool_per2')
             #.batch_normalization( name='bn_per2')
             .prelu(name='relu_per2'))

        (self.feed('relu2', 
                   'relu_per2')
             .concat(3, name='concat')
             .conv(3, 3, 64, 1, 1, relu=False, name='conv3')
             .max_pool(3, 3, 2, 2, name='pool3')
             #.batch_normalization(name='bn3')
             .prelu(name='relu3')
             .conv(3, 3, 64, 1, 1, relu=False, name='conv4')
             .max_pool(3, 3, 2, 2, name='pool4')
             #.batch_normalization( name='bn4')
             .prelu(name='relu4')
             .conv(3, 3, 128, 1, 1, relu=False, name='conv5')
             .max_pool(3, 3, 2, 2, name='pool5')
             #.batch_normalization( name='bn5')
             .prelu(name='relu5')
             .fc(256, relu=False, name='ip4')
             .prelu(name='reluip4')
             .fc(3, relu=False, name='ip6')
             .prelu(name='reluip6'))

        (self.feed('reluip4')
             .fc(3, relu=False, name='ip8')
             .prelu(name='reluip8'))

    def single_image():
        return False

    def position_tensor(self):
#hmm.... There is another prelu here...  we skipped it.
        return self.layers["reluip6"]

    def rotation_tensor(self):
        return self.layers["reluip8"]
