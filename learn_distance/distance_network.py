import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from robosims.unity import UnityGame
from util.util import *

class Distance_Model:
    def __init__(self, conf, cls, cheat=False, trainable=False):
        self.phase = tf.placeholder(tf.bool, name='phase')
        if cheat:
            cheat_distance = tf.placeholder(tf.float32, shape=[None, 1], name='cheat_distance')
        else:
            cheat_distance = None

        self.max_delta = math.sqrt(3 * conf.max_distance_delta * conf.max_distance_delta)
        self.mid_loss = 0.5 * self.max_delta * self.max_delta

        self.network = Distance_Network(conf, cls, "main", cheat_distance, trainable=trainable)
        self.pred_distance = self.network.get_output()

        # Mean squared error
        self.distance = tf.placeholder(tf.float32, name='distance', shape=[None, 1])
        if conf.loss_clip_min != None:
            clip_value_min = conf.loss_clip_min * self.distance
            clip_value_max = conf.loss_clip_max * self.distance
            self.delta = tf.clip_by_value(tf.abs(self.pred_distance - self.distance), clip_value_min, clip_value_max, name='clipped_delta') - clip_value_min
            # variable_summaries(self.delta)
        else:
            self.delta = self.pred_distance - self.distance

        self.loss = tf.nn.l2_loss(self.delta, name='loss')

        variable_summaries(self.loss)
        self.summary = tf.summary.merge_all()

        print("max distance is {} chance loss is {}".format(self.max_delta, self.chance_loss()))

    def summary_tensor(self):
        return self.summary

    def phase_tensor(self):
        return(self.phase)

    def pred_tensor(self):
        return self.pred_distance

    def true_tensor(self):
        return(self.distance)

    def loss_tensor(self):
        return(self.loss)

    def chance_loss(self):
        return self.mid_loss

    def true_value(self, env):
        return(np.reshape(env.distance(), [1]))

    def cheat_value(self, env):
        return 2 * self.true_value(env)

    def accuracy(self, true_distance, pred_distance):
        print(pred_distance.shape)
        print(true_distance.shape)

        return mape_accuracy(true_distance, pred_distance)

    def error_str(self, true_distance, pred_distance):
        err = mape(true_distance, pred_distance)
        err_pct_string = str(round(err, 2) *100) + "%"

        return "pred error " + err_pct_string

    def name(self):
        return "distance"

class Distance_Network:
    def __init__(self, conf, cls, scope, cheat = None, trainable=False):
        self.scope = scope

        with tf.variable_scope(scope):
            #Input and visual encoding layers

            self.s_input = tf.placeholder(shape=[None,conf.v_size,conf.h_size,conf.channels],dtype=tf.float32, name="s_input")
            self.t_input = tf.placeholder(shape=[None,conf.v_size,conf.h_size,conf.channels],dtype=tf.float32, name="t_input")

            with tf.variable_scope("siamese_network"):
                self.source_net = cls({'data': self.s_input}, trainable=trainable)

            with tf.variable_scope("siamese_network", reuse=True):
                self.target_net = cls({'data': self.t_input}, trainable=trainable)

            self.s_out = flatten(self.source_net.get_output())
            self.t_out = flatten(self.target_net.get_output())
            # self.s_out = self.source_net.get_output()
            # self.t_out = self.target_net.get_output()
              
            if cheat is not None:
                print("Distance network cheating....")
                combined = tf.concat(values=[self.t_out, self.s_out, cheat], axis=1)
            else:
                combined = tf.concat(values=[self.t_out, self.s_out], axis=1)

            hidden = slim.fully_connected(combined, 1024, activation_fn=tf.nn.elu,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None, scope='hidden_vector')

            self.distance_pred = slim.fully_connected(hidden,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None, scope='distance_pred')
            
    def get_output(self):
        return(self.distance_pred)

    def load(self, data_path, session, ignore_missing=False):
        with tf.variable_scope(self.scope):
            with tf.variable_scope("siamese_network"):
                self.source_net.load(data_path, session, ignore_missing)
