import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from robosims.unity import UnityGame
from util.util import *

class Direction_Model:
    def __init__(self, conf, cls, cheat=False, trainable=False):
        if cheat:
            self.cheat_direction = tf.placeholder(tf.float32, shape=[None, 3], name='cheat_direction')
        else:
            self.cheat_direction = None

        self.self.network = Direction_Network(conf, cls, "main", self.cheat_direction, trainable=trainable)
        self.pred_direction = self.network.get_output()

        self.mid_loss = 0.5 * 3 * 0.5 * conf.max_distance_delta * conf.max_distance_delta
        self.max_delta = conf.max_distance_delta

        # Mean squared error
        self.direction = tf.placeholder(tf.float32, name='direction', shape=[None, 3])
        if conf.loss_clip_min != None:
            clip_value_min = conf.loss_clip_min * self.direction
            clip_value_max = conf.loss_clip_max * self.direction
            self.delta = tf.clip_by_value(tf.abs(self.pred_direction - self.direction), clip_value_min, clip_value_max, name='clipped_delta') - clip_value_min
            # delta is not a scalar
            # variable_summaries(self.delta)
        else:
            self.delta = self.pred_direction - self.direction

        self.loss = tf.nn.l2_loss(self.delta, name='loss')

        variable_summaries(self.loss)
        self.summary = tf.summary.merge_all()

    def summary_tensor(self):
        return self.summary

    def pred_tensor(self):
        return self.pred_direction

    def true_tensor(self):
        return(self.direction)

    def loss_tensor(self):
        return(self.loss)

    def chance_loss(self):
        return self.mid_loss

    def true_value(self, env):
        return(np.reshape(env.direction(), [1,3]))

    def cheat_value(self, env):
        return 2 * self.true_value(env)

    def cheat_tensor(self):
        return self.cheat_direction

    def error_str(self, env, pred_direction):
        if pred_direction.size != 3:
            raise ValueError("error_str excpects pred_value to be of size 3")

        true_direction = env.direction()
        delta = true_direction - pred_direction

        s = ""
        for i in range(0,3):
            if abs(true_direction[i]) < 0.01:
                s += str(round(delta[i], 2)) + "/" + str(round(true_direction[i], 2))
            else:
                s += str(round(delta[i]/true_direction[i], 2) *100) + "%"
            if i != 2:
                s += ','

        return "pred error " + s

    def name(self):
        return "direction"

class Direction_Network():
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
                print("Direction network cheating....")
                combined = tf.concat(values=[self.t_out, self.s_out, cheat], axis=1)
                # combined = tf.concat(values=[cheat], axis=1)
            else:
                combined = tf.concat(values=[self.t_out, self.s_out], axis=1)

            hidden = slim.fully_connected(combined, 1024, activation_fn=tf.nn.elu,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None, scope='hidden_vector')

            self.direction_pred = slim.fully_connected(hidden,3,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None, scope='direction_pred')

            
    def get_output(self):
        return(self.direction_pred)

    def load(self, data_path, session, ignore_missing=False):
        with tf.variable_scope(self.scope):
            with tf.variable_scope("source"):
                self.source_net.load(data_path, session, ignore_missing)
            with tf.variable_scope("target"):
                self.target_net.load(data_path, session, ignore_missing)
