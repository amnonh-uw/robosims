import os
import sys
import numpy as np
import scipy.misc
import tensorflow as tf
import math
import util
from util.util import *

sfmdir = "/home/amnonh/robosims/networks/SfMLearner"
sys.path.insert(0, sfmdir)

from SfMLearner import SfMLearner


class sfm_depth:
    tf_trainable = False
    tf_testable = False
    img_height = 320
    img_width = 416
    mode = 'both'

    def predictor(self, t, s, model):
        self.predict_counter += 1
        self.directory = 'output/' + str(self.predict_counter)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.raw_h = t.shape[0]
        self.raw_w = t.shape[1]

        zoom_y = self.img_height/self.raw_h
        zoom_x = self.img_width/self.raw_w

        if zoom_x != 1 or zoom_y != 1:
            s = scipy.misc.imresize(s, (self.img_height, self.img_width))
            t = scipy.misc.imresize(t, (self.img_height, self.img_width))

        save_image(self.directory + '/' + "resized_source" + '.png', s)
        save_image(self.directory + '/' + "resized_target" + '.png', s)

        if self.mode == 'depth':
            fetches = self.fetch_depth(t, s, model)

        if self.mode == 'pose':
            fetches = self.fetch_both(t, s, model)

        if self.mode == 'both':
            fetches = self.fetch_both(t, s, model)

        for key in fetches:
            print(key)
            if "image" in key or "depth" in key or "disp" in key:
                im_batch = fetches[key]
                im = np.squeeze(im_batch, axis=0)
                if im.shape[2] == 1:
                    im = np.squeeze(im, axis=2)
                save_image(self.directory + '/' + key + '.png', im)
            else:
                print(key, fetches[key])

        if 'pose' in fetches:
            pred = fetches['pose']
            pred_pose = pred[0, :, :]
            print('pred_pose', pred_pose)
            return pred_pose
        else:
            return np.zeros([1,6])

    def __init__(self, conf):
        self.sfm = SfMLearner()
        self.sfm.setup_inference(self.img_height,
                    self.img_width,
                    mode=self.mode,
                    seq_length=2)

        self.load(conf)
        self.vert_fov = conf.vert_fov
        self.near_clip_plane = conf.near_clip_plane
        self.predict_counter = 0

    def load(self, conf):
        print('loading from ' + conf.checkpoint_dir)
        ckpt_file = tf.train.latest_checkpoint(conf.checkpoint_dir)
        saver = tf.train.Saver() 
        self.sess =  tf.Session()
        saver.restore(self.sess, ckpt_file)

    def fetch_both(self, t, s, model):
        # inference assumes that the middle image is the target
        # so we are transfering two images, the first is the target
        # second is the source

        seq = concat_images(t, s)
        seq = np.expand_dims(seq, axis=0)

        return =  self.sfm.inference(seq, self.sess, mode=self.mode)

    def fetch_depth(self, t, s, model):
        t = np.expand_dims(t, axis=0)

        return self.sfm.inference(t, self.sess, mode=self.mode)
