import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys

examples_dir = '/home/amnonh/demon/examples/'
weights_dir = os.path.join(examples_dir,'..','weights')
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))

from depthmotionnet.networks_original import *


class demon:
    tf_trainable = False
    tf_testable = False

    def __init__(self):
        gpu_options = tf.GPUOptions()
        gpu_options.per_process_gpu_memory_fraction=0.8
        self.session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

        if tf.test.is_gpu_available(True):
            self.data_format='channels_first'
        else: # running on cpu requires channels_last data format
            self.data_format='channels_last'

        # init networks
        self.bootstrap_net = BootstrapNet(self.session, self.data_format)
        self.iterative_net = IterativeNet(self.session, self.data_format)
        self.refine_net = RefinementNet(self.session, self.data_format)

        self.session.run(tf.global_variables_initializer())

        # load weights
        saver = tf.train.Saver()
        saver.restore(self.session,os.path.join(weights_dir,'demon_original'))

    def predictor(self, t, s, model):
        im1 = Image.fromarray(s, 'RGB')
        im2 = Image.fromarray(s, 'RGB')
        result = self.eval(im1, im2)
        print("predicted translation ", self.translation)
        print("predicted rotation ", self.rotation)
        return self.translation

    def prepare_input_data(self, img1, img2, data_format):
        """Creates the arrays used as input from the two images."""
        # scale images if necessary
        if img1.size[0] != 256 or img1.size[1] != 192:
            img1 = img1.resize((256,192))
        if img2.size[0] != 256 or img2.size[1] != 192:
            img2 = img2.resize((256,192))
        img2_2 = img2.resize((64,48))
        
        # transform range from [0,255] to [-0.5,0.5]
        img1_arr = np.array(img1).astype(np.float32)/255 -0.5
        img2_arr = np.array(img2).astype(np.float32)/255 -0.5
        img2_2_arr = np.array(img2_2).astype(np.float32)/255 -0.5
    
        if data_format == 'channels_first':
            img1_arr = img1_arr.transpose([2,0,1])
            img2_arr = img2_arr.transpose([2,0,1])
            img2_2_arr = img2_2_arr.transpose([2,0,1])
            image_pair = np.concatenate((img1_arr,img2_arr), axis=0)
        else:
            image_pair = np.concatenate((img1_arr,img2_arr),axis=-1)
    
        result = {
            'image_pair': image_pair[np.newaxis,:],
            'image1': img1_arr[np.newaxis,:], # first image
            'image2_2': img2_2_arr[np.newaxis,:], # second image with (w=64,h=48)
        }
        return result

    def eval(self, img1, img2):
        # 
        # DeMoN has been trained for specific internal camera parameters.
        #
        # If you use your own images try to adapt the intrinsics by cropping
        # to match the following normalized intrinsics:
        #
        #  K = (0.89115971  0           0.5)
        #      (0           1.18821287  0.5)
        #      (0           0           1  ),
        #  where K(1,1), K(2,2) are the focal lengths for x and y direction.
        #  and (K(1,3), K(2,3)) is the principal point.
        #  The parameters are normalized such that the image height and width is 1.
        #

        input_data = self.prepare_input_data(img1,img2,self.data_format)

        # run the network
        result = self.bootstrap_net.eval(input_data['image_pair'], input_data['image2_2'])
        for i in range(3):
            result = self.iterative_net.eval(
                input_data['image_pair'], 
                input_data['image2_2'], 
                result['predict_depth2'], 
                result['predict_normal2'], 
                result['predict_rotation'], 
                result['predict_translation']
            )

        self.rotation = result['predict_rotation'], 
        self.translation = result['predict_translation']

        result = self.refine_net.eval(input_data['image1'],result['predict_depth2'])
        return result
