import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from robosims.unity import UnityGame
from util.util import *

class Distance_Model:
    def __init__(self, conf, cls, cheat=False, trainable=False):
        if cheat:
            cheat_distance = tf.placeholder(tf.float32, shape=[None, 1], name='cheat_distance')
        else:
            cheat_distance = None

        self.network = Distance_Network(conf, cls, "main", cheat_distance, trainable=trainable)
        self.pred_distance = self.network.get_output()

        # Mean squared error
        self.distance = tf.placeholder(tf.float32, name='distance', shape=[None, 1])
        self.l2_loss = tf.nn.l2_loss(self.pred_distance - self.distance, name='loss')

        max_distance = math.sqrt(3 * conf.max_distance_delta * conf.max_distance_delta)
        self.mid_loss = 0.5 * max_distance * max_distance
        self.max_loss = max_distance * max_distance

        if conf.loss_clip_min != None:
            variable_summaries(self.l2_loss)
            clip_value_min = self.max_loss * conf.loss_clip_min
            clip_value_max = 9999999999
            print("clipping between {} and {}".format(clip_value_min, clip_value_max))
            self.loss = tf.clip_by_value(self.l2_loss, clip_value_min, clip_value_max, name='loss') - clip_value_min
        else:
            self.loss = self.l2_loss

        variable_summaries(self.loss)
        self.summary = tf.summary.merge_all()

        print("max distance is {} chance loss is {}".format(max_distance, self.chance_loss()))

    def summary_tensor(self):
        return self.summary

    def pred_tensor(self):
        return self.pred_distance

    def true_tensor(self):
        return(self.distance)

    def loss_tensor(self):
        return(self.loss)

    def chance_loss(self):
        return self.mid_loss

    def max_loss(self):
        return self.max_loss

    def true_value(self, env):
        return(np.reshape(env.distance(), [1,1]))

    def cheat_value(self, env):
        return 2 * self.true_value(env)

    def error_str(self, env, pred_distance):
        if pred_distance.size != 1:
            raise ValueError("accuracy excpects pred_value to be of size 1")
        if isinstance(pred_distance, np.ndarray):
            pred_distance = pred_distance[0,0]

        true_distance = env.distance()
        delta = true_distance - pred_distance

        err_pct_string = str(round(delta/true_distance, 2) *100) + "%"
        true_distance = round(true_distance, 2)
        delta = round(delta, 2)
        err_frac_string = str(delta) + "/" + str(true_distance)

        return "error " + err_frac_string + " " + err_pct_string

    def name(self):
        return "distance"

class Distance_Network:
    def __init__(self, conf, cls, scope, cheat = None, trainable=False):
        self.scope = scope

        with tf.variable_scope(scope):
            #Input and visual encoding layers

            self.s_input = tf.placeholder(shape=[None,conf.v_size,conf.h_size,conf.channels],dtype=tf.float32, name="s_input")
            self.t_input = tf.placeholder(shape=[None,conf.v_size,conf.h_size,conf.channels],dtype=tf.float32, name="t_input")

            with tf.variable_scope("source"):
                self.source_net = cls({'data': self.s_input}, trainable=trainable)

            with tf.variable_scope("target"):
                self.target_net = cls({'data': self.t_input}, trainable=trainable)

            self.s_out = flatten(self.source_net.get_output())
            self.t_out = flatten(self.target_net.get_output())
              
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
            with tf.variable_scope("source"):
                self.source_net.load(data_path, session, ignore_missing)
            with tf.variable_scope("target"):
                self.target_net.load(data_path, session, ignore_missing)
