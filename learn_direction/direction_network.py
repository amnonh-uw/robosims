import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from robosims.unity import UnityGame

class Direction_Model:
    def __init__(self, args, cls, cheat=False, trainable=False):
        if cheat:
            cheat_direction = tf.placeholder(tf.float32, shape=[None, 3], name='cheat_direction')
        else:
            cheat_direction = None
        self.network = Direction_Network(args, cls, "main", cheat_direction, trainable=trainable)
        self.pred_direction = self.network.get_output()

        # Mean squared error
        self.direction = tf.placeholder(tf.float32, name='direction', shape=[None, 3])
        self.loss = tf.nn.l2_loss(self.pred_direction - self.direction, name='loss')
        self.mid_loss = 3 * 0.5 * args.max_distance_delta * args.max_distance_delta

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

    def error_str(self, env, pred_direction):
        if pred_direction.size != 3:
            raise ValueError("accuracy excpects pred_value to be of size 1")
        if isinstance(pred_direction, np.ndarray):
            pred_direction = pred_direction[0, :]

        true_direction = env.direction()
        delta = true_direction - pred_direction

        s = ""
        for i in range(0,3):
            if true_direction[i] < 0.01:
                s += str(round(delta[i], 2)) + "/" + str(round(true_direction[i], 2))
            else:
                s += str(round(delta[i]/true_direction[i], 2) *100) + "%"
            if i != 2:
                s += ','

        return "error " + s

    def name(self):
        return "direction"

class Direction_Network():
    def __init__(self, args, cls, scope, cheat = None, trainable=False):
        self.scope = scope

        with tf.variable_scope(scope):
            #Input and visual encoding layers

            self.s_input = tf.placeholder(shape=[None,args.v_size,args.h_size,args.channels],dtype=tf.float32, name="s_input")
            self.t_input = tf.placeholder(shape=[None,args.v_size,args.h_size,args.channels],dtype=tf.float32, name="t_input")

            with tf.variable_scope("source"):
                self.source_net = cls({'data': self.s_input}, trainable=trainable)

            with tf.variable_scope("target"):
                self.target_net = cls({'data': self.t_input}, trainable=trainable)

            self.s_out = self.flatten(self.source_net.get_output())
            self.t_out = self.flatten(self.target_net.get_output())
              
            if cheat is not None:
                print("Direction network cheating....")
                combined = tf.concat(values=[self.t_out, self.s_out, cheat], axis=1)
                # combined = tf.concat(values=[cheat], axis=1)
            else:
                combined = tf.concat(values=[self.t_out, self.s_out], axis=1)

            # self.s_conv_layers = self.flatten(self.source_net.conv_layers())
            # self.t_conv_layers = self.flatten(self.target_net.conv_layers())

            # combined = tf.concat(values=[self.t_conv_layers, self.s_conv_layers], axis=1)

            hidden = slim.fully_connected(combined, 1024, activation_fn=tf.nn.elu,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None, scope='hidden')

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

    def flatten(self, t, max_size=0):
        if isinstance(t, list):
            l = []
            for t_item in t:
                l.append(self.flatten(t_item, max_size))

            return l
            
        input_shape = t.get_shape()
        if input_shape.ndims == 4:
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            print("flattening " + t.name + " size " + str(dim))
            t = tf.reshape(t, [-1, dim])
        elif input_shape.ndims == 1:
            dim = input_shape[1]
        else:
            raise ValueError("invalid number of dimentions " + str(input_shape.ndims))

        if max_size != 0 and dim > max_size:
            print("size too large, inserting a hidden layer")
            # insert a fully connected layer
            hidden = slim.fully_connected(t, max_size,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)

            return hidden
        else:
            return t

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        print("Initializer called for shape {}".format(shape))
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        print("mean of output {}".format(np.mean(out)))
        print("max of output {}".format(np.max(out)))
        return tf.constant(out, dtype=tf.float32, shape=shape)
    return _initializer
