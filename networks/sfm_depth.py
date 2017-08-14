import os
import sys
import numpy as np
import scipy.misc
import tensorflow as tf

sfmdir = "/home/amnonh/SfMLearner"
sys.path.insert(0, sfmdir)

from SfMLearner import SfMLearner
from utils import *


class sfm_depth:
    tf_trainable = False
    tf_testable = False
    img_height = 300
    img_width = 400
    mode = 'pose'

    def __init__(self, conf):
        self.sfm = SfMLearner()
        self.sfm.setup_inference(self.img_height,
                    self.img_width,
                    mode=self.mode,
                    seq_length=2)

        self.load(conf)

    def load(self, conf):
        ckpt_file = tf.train.latest_checkpoint(conf.checkpoint_dir)
        saver = tf.train.Saver([var for var in tf.trainable_variables()]) 
        self.sess =  tf.Session()
        saver.restore(self.sess, ckpt_file)

    def predictor(self, t, s, model):
        # inference assumes that the middle image is the target
        # so we are transfering two images, the first is the source and the
        # second is the target
        seq = np.hstack((t, s))
        pred = self.sfm.inference(seq[None,:,:,:], self.sess, mode=self.mode)
        pred = pred['pose']

        pred_pose = pred[0, :, :]
        # units = 0.001154661
        # pred_pose[0, 0] /= units
        # pred_pose[0, 1] /= units
        # pred_pose[0, 2] /= units
        print('pred_pose', pred_pose)
        return pred_pose
