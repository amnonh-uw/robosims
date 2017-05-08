import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from robosims.unity import UnityGame
from util.util import *
from scipy.spatial.distance import euclidean
from numpy.linalg import norm

class Translation_Model:
    def __init__(self, conf, cls, cheat=False, trainable=False):
        if cheat:
            self.cheat_translation = tf.placeholder(tf.float32, shape=[None, 3], name='cheat_translation')
        else:
            self.cheat_translation = None

        self.network = Translation_Network(conf, cls, "main", self.cheat_translation, trainable=trainable)
        self.pred_translation = self.network.get_output()

        self.mid_loss = 0.5 * 3 * 0.5 * conf.max_distance_delta * conf.max_distance_delta
        self.max_delta = conf.max_distance_delta

        # Mean squared error
        self.translation = tf.placeholder(tf.float32, name='translation', shape=[None, 3])
        if conf.loss_clip_min != None:
            clip_value_min = conf.loss_clip_min * self.translation
            clip_value_max = conf.loss_clip_max * self.translation
            self.delta = tf.clip_by_value(tf.abs(self.pred_translation - self.translation), clip_value_min, clip_value_max, name='clipped_delta') - clip_value_min
            # delta is not a scalar
            # variable_summaries(self.delta)
        else:
            self.delta = self.pred_translation - self.translation

        self.loss = tf.nn.l2_loss(self.delta, name='loss')

        variable_summaries(self.loss)
        self.summary = tf.summary.merge_all()

    def summary_tensor(self):
        return self.summary

    def pred_tensor(self):
        return self.pred_translation

    def true_tensor(self):
        return(self.translation)

    def loss_tensor(self):
        return(self.loss)

    def chance_loss(self):
        return self.mid_loss

    def true_value(self, env):
        return(np.reshape(env.translation(), [1,3]))

    def cheat_value(self, env):
        return 2 * self.true_value(env)

    def cheat_tensor(self):
        return self.cheat_translation

    def accuracy(self, env, pred_translation):
        pred_transaction = as_vecotr(pred_translation, 3)
        true_translation = env.translation()

        a = np.zeros(3)
        for i in range(0, 3):
            a[i] = mape_accuracy(true_translation[i], pred_translation[i])

        return(np.min(a[i]))

    def error_str(self, env, pred_translation):
        pred_transaction = as_vector(pred_translation, 3)

        true_translation = env.translation()
        delta = true_translation - pred_translation

        s = "pred_error "
        for i in range(0,3):
            err = mape(true_translation[i], pred_translation[i])
            s += str(round(err, 2) *100) + "%"
            if i != 2:
                s += ','

        return s

    def name(self):
        return "translation"

class Translation_Network():
    def __init__(self, conf, cls, scope, cheat = None, trainable=False):
        self.scope = scope

        with tf.variable_scope(scope):
            #Input and visual encoding layers

            self.s_input = tf.placeholder(shape=[None,conf.v_size,conf.h_size,conf.channels],dtype=tf.float32, name="s_input")
            self.t_input = tf.placeholder(shape=[None,conf.v_size,conf.h_size,conf.channels],dtype=tf.float32, name="t_input")

            if cls.single_image():
                with tf.variable_scope("siamese_network"):
                    self.source_net = cls({'data': self.s_input}, trainable=trainable)

                with tf.variable_scope("siamese_network", reuse=True):
                    self.target_net = cls({'data': self.t_input}, trainable=trainable)

                self.s_out = flatten(self.source_net.get_output())
                self.t_out = flatten(self.target_net.get_output())
              
                if cheat is not None:
                    print("Translation network cheating....")
                    combined = tf.concat(values=[self.t_out, self.s_out, cheat], axis=1)
                else:
                     combined = tf.concat(values=[self.t_out, self.s_out], axis=1)

                hidden = slim.fully_connected(combined, 1024, activation_fn=tf.nn.elu,
                    weights_initializer=normalized_columns_initializer(1.0),
                    biases_initializer=None, scope='hidden_vector')

                self.translation_pred = slim.fully_connected(hidden,3,
                    activation_fn=None,
                    weights_initializer=normalized_columns_initializer(1.0),
                    biases_initializer=None, scope='translation_pred')
            else:
                self.net = cls({'image_data': self.s_input, 'image_data_pert' : self.t_input}, trainable=trainable)
                self.translation_pred = self.net.position_tensor()

            
    def get_output(self):
        return(self.translation_pred)

    def load(self, data_path, session, ignore_missing=False):
        with tf.variable_scope(self.scope):
            with tf.variable_scope("siamese_network"):
                self.source_net.load(data_path, session, ignore_missing)
