import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from robosims.unity import UnityGame
from util.util import *

class Translation_Model:
    def __init__(self, conf, cls, cheat=False, trainable=True):
        self.relative_errors = conf.relative_errors
        self.clip_loss_lambda = conf.clip_loss_lambda
        self.pose_dims = conf.pose_dims
        self.highlight_rot_absolute_error = conf.highlight_rot_absolute_error
        self.highlight_rot_relative_error = conf.highlight_rot_relative_error
        self.highlight_pos_absolute_error = conf.highlight_pos_absolute_error
        self.highlight_pos_relative_error = conf.highlight_pos_relative_error

        self.phase = tf.placeholder(tf.bool, name='phase')
        if cheat:
            self.cheat = tf.placeholder(tf.float32, shape=[None, self.pose_dims], name='cheat')
        else:
            self.cheat = None


        self.network = Translation_Network(conf, cls, "main", self.phase, self.cheat, trainable=trainable)
        self.pred_translation = self.network.get_output()
        self.translation = tf.placeholder(tf.float32, name='translation', shape=[None, self.pose_dims])

        pos_dims = min(self.pose_dims, 3)
        rot_dims = max(self.pose_dims - 3, 0)

        pos_truth = tf.slice(self.translation, [0, 0], [-1, pos_dims])
        pos_pred = tf.slice(self.pred_translation, [0, 0], [-1, pos_dims])
        pos_error = self.compute_error(pos_pred, pos_truth)
        pos_loss = self.compute_loss(pos_error, pos_truth, 'pos_loss')

        variable_summaries(pos_loss)

        if rot_dims != 0:
            rot_truth = tf.slice(self.translation, [0, 3], [-1, rot_dims])
            rot_pred = tf.slice(self.pred_translation, [0, 3], [-1, rot_dims])
            rot_error = self.compute_error(rot_pred, rot_truth)
            rot_loss = self.compute_loss(rot_error, rot_truth, 'rot_loss')

            variable_summaries(rot_loss)
    
            self.error = tf.concat([pos_error, rot_error], 1)
            self.loss = pos_loss + rot_loss * conf.rot_loss_factor
            self.detailed_loss = tf.stack([pos_loss, rot_loss])
        else:
            self.error = pos_error
            self.loss = pos_loss
            self.detailed_loss = pos_loss

        variable_summaries(self.loss)
        self.summary = tf.summary.merge_all()

    def compute_loss(self, error, truth, name):
        if self.clip_loss_lambda == None:
            return tf.nn.l2_loss(error, name=name)
        else:
            return tf.reduce_sum(tf.maximum(0., error*error - conf.clip_loss_lambda*truth*truth), name=name)

    def compute_error(self, pred, truth):
        error = pred - truth
        if self.relative_errors:
            return tf.abs(tf.divide(pos_error,  pos_truth+0.001))
        else:
            return error

    def summary_tensor(self):
        return self.summary

    def phase_tensor(self):
        return(self.phase)

    def pred_tensor(self):
        return self.pred_translation

    def error_tensor(self):
        return(self.error)

    def true_tensor(self):
        return(self.translation)

    def loss_tensor(self):
        return(self.loss)

    def detailed_loss_tensor(self):
        return(self.detailed_loss)

    def rescale_value(self, dims):
        rescale = [1, 1, 1, 0.5]
        return(rescale[:dims])

    def true_value(self, env):
        return np.reshape(env.translation(dims=self.pose_dims), [self.pose_dims])

    def cheat_value(self, env):
        return self.true_value(env)

    def cheat_tensor(self):
        return self.cheat

    def error_strings(self, true_translation, pred_translation):
        if true_translation.shape[0] != 1:
            raise ValueError("error_str excpects test_transaltion to be a vector")

        strings = []
        colors = []

        texts = ["x {} err:", "y {} err:", "z {} err:", "r {} err:"]
        for i in range(0,self.pose_dims):
            if i <= 3:
                highlight_absolute_error = self.highlight_pos_absolute_error
                highlight_relative_error = self.highlight_pos_relative_error
            else:
                highlight_absolute_error = self.highlight_rot_absolute_error
                highlight_relative_error = self.highlight_rot_relative_error

            relative_err = map_error(true_translation[0,i], pred_translation[0,i])
            absolute_err = abs_error(true_translation[0,i], pred_translation[0,i])

            color = "white"
            if absolute_err > highlight_absolute_error:
                if relative_err > highlight_relative_error:
                    color = "red"

            s = texts[i].format(pred_translation[0,i])
            s += str(round(relative_err, 2) *100) + "% "
            s += '('
            s += str(round(absolute_err, 2))
            s += "/"
            s += str(round(true_translation[0,i], 2))
            s += ')'
            strings.append(s)
            colors.append(color)

        return strings, colors

    def take_prediction_step(self, env, pred_value):
        env.take_prediction_step(pred_value)

    def name(self):
        return "translation"

class Translation_Network():
    def __init__(self, conf, cls, scope, phase, cheat = None, trainable=True, pose_dims=3):
        self.scope = scope
        self.pose_dims = conf.pose_dims
        self.single_image = cls.single_image()

        with tf.variable_scope(scope):
            input_shape = [None] + conf.image_shape
            self.s_input = tf.placeholder(shape=input_shape,dtype=tf.float32, name="s_input")
            self.t_input = tf.placeholder(shape=input_shape,dtype=tf.float32, name="t_input")

            if self.single_image:
                with tf.variable_scope("siamese_network"):
                    self.source_net = cls({'data': self.s_input}, phase, trainable=trainable)

                with tf.variable_scope("siamese_network", reuse=True):
                    self.target_net = cls({'data': self.t_input}, phase, trainable=trainable)

                self.s_out = flatten(self.source_net.get_output())
                self.t_out = flatten(self.target_net.get_output())
              
                if cheat is not None:
                    print("Translation network cheating....")
                    combined = tf.concat(values=[self.t_out, self.s_out, cheat], axis=1)
                else:
                     combined = tf.concat(values=[self.t_out, self.s_out], axis=1)
            else:
                self.net = cls({'data1': self.s_input, 'data2' : self.t_input}, phase, trainable=trainable)
                if cheat is not None:
                    print("Translation network cheating....")
                    combined = tf.concat(values=[self.t_out, self.s_out, cheat], axis=1)
                else:
                    combined = flatten(self.net.get_output())

            hidden = slim.fully_connected(combined, 1024,
                    activation_fn=tf.nn.relu,
                    weights_initializer=normalized_columns_initializer(1.0),
                    biases_initializer=None, scope='hidden_vector')

            self.translation_pred = slim.fully_connected(hidden,self.pose_dims,
                    # can't use relu as activation funtion here - we want negative values!
                    activation_fn=None,
                    weights_initializer=normalized_columns_initializer(1.0),
                    biases_initializer=None, scope='translation_pred')

    def get_output(self):
        return(self.translation_pred)

    def load(self, data_path, session, ignore_missing=False):
        with tf.variable_scope(self.scope):
            if self.single_image:
                with tf.variable_scope("siamese_network"):
                    self.source_net.load(data_path, session, ignore_missing)
            else:
                self.net.load(data_path, session, ignore_missing)
