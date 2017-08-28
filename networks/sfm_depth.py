import os
import sys
import numpy as np
import scipy.misc
import tensorflow as tf
import math

sfmdir = "/home/amnonh/SfMLearner"
sys.path.insert(0, sfmdir)

from SfMLearner import SfMLearner
from utils import *


class sfm_depth:
    tf_trainable = False
    tf_testable = False
    img_height = 320
    img_width = 416
    mode = 'all'

    def __init__(self, conf):
        self.sfm = SfMLearner()
        self.sfm.setup_inference(self.img_height,
                    self.img_width,
                    mode=self.mode,
                    seq_length=2)

        self.load(conf)
        self.vert_fov = conf.vert_fov
        self.near_clip_plane = self.near_clip_plane
        self.predict_counter = 0

    def load(self, conf):
        print('loading from ' + conf.checkpoint_dir)
        ckpt_file = tf.train.latest_checkpoint(conf.checkpoint_dir)
        saver = tf.train.Saver() 
        self.sess =  tf.Session()
        saver.restore(self.sess, ckpt_file)

    def image_resize(self, im):
        if self.img_height != self.raw_h or self.img_width != self.raw_w:
            im = scipy.misc.imresize(im, (self.raw_h, self.raw_w))

        return im
        
    def predictor(self, t, s, model):
        # inference assumes that the middle image is the target
        # so we are transfering two images, the first is the source and the
        # second is the target
        self.raw_h = t.shape[0]
        self.raw_w = t.shape[1]

        zoom_y = self.img_height/self.raw_h
        zoom_x = self.img_width/self.raw_w

        aspect_ratio = float(self.raw_w) / float(self.raw_h)
        vert_fov = math.radians(self.vert_fov)
        near_clip_plane = self.near_clip_plane
        horz_fov =  2 * math.atan(math.tan(vert_fovr / 2) * aspect_ratio)

        fx = near_clip_plane
        fy = near_clip_plane
        cx = fx * math.tan(0.5 * horz_fov)
        cy = fy * math.tan(0.5 * vert_fov)

        if zoom_x != 1 or zoom_y != 1:
            s = scipy.misc.imresize(s, (self.img_height, self.img_width))
            t = scipy.misc.imresize(t, (self.img_height, self.img_width))
            fx *= zoom_x
            cx *= zoom_x
            fy *= zoom_y
            cy *= zoom_y
        
        raw_mat_vecs = np.array([[fx,0.,cx,0.,fy,cy,0.,0.,1.]])

        seq = np.hstack((t, s))
        self.predict_counter += 1
        directory = 'output/' + str(self.predict_counter)
        if not os.path.exists(directory):
            os.makedirs(directory)

        fetches = self.sfm.inference([seq[None,:,:,:], raw_mat_vecs], self.sess, mode=self.mode)
        for key in fetches:
            if "image" in key or "exp_mask" in key or "proj_error" in key:
                im_batch = fetches[key]
                im = np.squeeze(im_batch, axis=0)
                if im.shape[2] == 1:
                    im = np.squeeze(im, axis=2)
                scipy.misc.imsave(directory + '/' + key + '.png', im)
            else:
                print(key, fetches[key])

        pred = fetches['pose']

        pred_pose = pred[0, :, :]
        units = 0.001154661
        pred_pose[0, 0] /= units
        pred_pose[0, 1] /= units
        pred_pose[0, 2] /= units
        print('pred_pose', pred_pose)
        return pred_pose

