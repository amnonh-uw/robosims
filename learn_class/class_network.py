import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from robosims.unity import UnityGame
from util.util import *

class Class_Model:
    def __init__(self, conf, cls, cheat=False, trainable=False):
        if cheat:
            self.cheat_class = tf.placeholder(tf.float32, shape=[None, 1], name='cheat_class')
        else:
            self.cheat_class = None

        self.network = Class_Network(conf, cls, "main", self.cheat_class, trainable=trainable)
        self.pred_logits = self.network.get_logits()
        self.pred_softmax = self.network.get_softmax()

        # cross entropy error
        self.label = tf.placeholder(tf.int32, name='label', shape=[None, 1])
        label_vec = tf.reshape(self.label, [-1])

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_vec,
                logits=self.pred_logits, name='loss')

        variable_summaries(tf.reshape(self.loss, []))
        self.summary = tf.summary.merge_all()

    def summary_tensor(self):
        return self.summary

    def pred_tensor(self):
        return self.pred_softmax

    def true_tensor(self):
        return(self.label)

    def loss_tensor(self):
        return(self.loss)

    def true_value(self, env):
        return(np.reshape(env.get_class(), [1, 1]))

    def cheat_value(self, env):
        return self.true_value(env)/10

    def cheat_tensor(self):
        return self.cheat_class

    def accuracy(self, env, pred_softmax):
        if pred_softmax.size != 6:
            raise ValueError("accuracy excpects pred_value to be of size 6")

        cls = np.argmax(pred_softmax)
        return cls == env.get_class()

    def error_str(self, env, pred_softmax):
        if pred_softmax.size != 6:
            raise ValueError("accuracy excpects pred_value to be of size 6")

        return "pred error " + str(self.accuracy(env, pred_softmax))

    def name(self):
        return "class"

class Class_Network:
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

            self.s_out = self.source_net.get_output()
            self.t_out = self.target_net.get_output()
              
            if cheat is not None:
                print("Class network cheating....")
                combined = tf.concat(values=[self.t_out, self.s_out, cheat], axis=1)
            else:
                combined = tf.concat(values=[self.t_out, self.s_out], axis=1)

            hidden = slim.fully_connected(combined, 1024, activation_fn=tf.nn.relu,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None, scope='hidden_vector')

            self.pred_logits = slim.fully_connected(hidden, 6,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None, scope='class_pred')

            self.pred_softmax = tf.nn.softmax(self.pred_logits, name='softmax')
            
    def get_logits(self):
        return self.pred_logits

    def get_softmax(self):
        return self.pred_softmax

    def load(self, data_path, session, ignore_missing=False):
        with tf.variable_scope(self.scope):
            with tf.variable_scope("siamese_network"):
                self.source_net.load(data_path, session, ignore_missing)