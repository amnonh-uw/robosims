import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from robosims.unity import UnityGame
from util.util import *

class Translation_Model:
    def __init__(self, conf, cls, cheat=False, trainable=True):
        self.relative_errors = conf.relative_errors
        self.phase = tf.placeholder(tf.bool, name='phase')
        if cheat:
            self.cheat_translation = tf.placeholder(tf.float32, shape=[None, 3], name='cheat_translation')
        else:
            self.cheat_translation = None

        self.network = Translation_Network(conf, cls, "main", self.phase, self.cheat_translation, trainable=trainable)
        self.pred_translation = self.network.get_output()

        self.translation = tf.placeholder(tf.float32, name='translation', shape=[None, 3])
        self.error = self.pred_translation - self.translation

        if self.relative_errors:
            self.error = tf.abs(tf.divide(self.error,  self.translation+0.001))
            if conf.error_clip_min != None:
                self.error = tf.clip_by_value(self.error, conf.error_clip_min,
                                conf.error_clip_max, name='clipped_error') - conf.error_clip_min

        # l2 loss
        self.l2_loss = tf.nn.l2_loss(self.error, name='l2_loss')
        variable_summaries(self.l2_loss)
        self.summary = tf.summary.merge_all()

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
        return(self.l2_loss)

    def true_value(self, env):
        return(np.reshape(env.translation(), [3]))

    def cheat_value(self, env):
        return self.true_value(env)

    def cheat_tensor(self):
        return self.cheat_translation

    def accuracy(self, true_translation, pred_translation):
        a = np.zeros(3, dtype=np.float32)

        if self.relative_errors:
            for i in range(0, 3):
                a[i] = map_accuracy(true_translation[:,i], pred_translation[:,i])
        else:
            for i in range(0, 3):
                a[i] = abs_accuracy(true_translation[:,i], pred_translation[:,i])

        return(np.min(a))

    def error_str(self, true_translation, pred_translation):
        if true_translation.shape[0] != 1:
            raise ValueError("error_str excpects test_transaltion to be a vector")

        s = "pred_error "
        for i in range(0,3):
            relative_err = map_error(true_translation[0,i], pred_translation[0,i])
            s += str(round(relative_err, 2) *100) + "% "
            absolute_err = abs_error(true_translation[0,i], pred_translation[0,i])
            s += str(round(absolute_err, 2))
            s += " from "
            s += str(round(true_translation[0,i], 2))
            if i != 2:
                s += ','

        return s

    def name(self):
        return "translation"

class Translation_Network():
    def __init__(self, conf, cls, scope, phase, cheat = None, trainable=True):
        self.scope = scope

        with tf.variable_scope(scope):
            #Input and visual encoding layers

            self.s_input = tf.placeholder(shape=[None,conf.v_size,conf.h_size,conf.channels],dtype=tf.float32, name="s_input")
            self.t_input = tf.placeholder(shape=[None,conf.v_size,conf.h_size,conf.channels],dtype=tf.float32, name="t_input")

            if cls.single_image():
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

                hidden = slim.fully_connected(combined, 1024,
                    activation_fn=None,
                    # activation_fn=tf.nn.elu,
                    weights_initializer=normalized_columns_initializer(1.0),
                    biases_initializer=None, scope='hidden_vector')

                self.translation_pred = slim.fully_connected(hidden,3,
                    activation_fn=None,
                    # activation_fn=tf.nn.elu,
                    weights_initializer=normalized_columns_initializer(1.0),
                    biases_initializer=None, scope='translation_pred')
            else:
                self.net = cls({'image_data': self.s_input, 'image_data_pert' : self.t_input}, phase, trainable=trainable)
                self.translation_pred = self.net.position_tensor()
                if cheat is not None:
                    zeros = tf.zeros([1,3])
                    print("Translation network cheating....")
                    combined = tf.concat(values=[zeros, cheat], axis=1)
                    self.translation_pred = slim.fully_connected(combined,3,
                        activation_fn=None,
                        weights_initializer=normalized_columns_initializer(1.0),
                         biases_initializer=None, scope='translation_pred')

    def get_output(self):
        return(self.translation_pred)

    def load(self, data_path, session, ignore_missing=False):
        with tf.variable_scope(self.scope):
            with tf.variable_scope("siamese_network"):
                self.source_net.load(data_path, session, ignore_missing)
