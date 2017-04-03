import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
from .posenet import GoogLeNet

# ### Actor-Critic Network

# because we are executing in continuous coordinates, and because in the real world condition change,
# we would never be able to get the exact target image required. What's a good distance measure to decide
# we are close enough (for camera images)?    Use tolerance from pose space, and in real world series of zero action

# resolution mismatch - camera vs googlenet trainnign image database?

# tranform color to black and white or use three channels?

# Run gradients on the same posenet network for both sides? How?

# talk to hugh - he has a pre-trained network structure for 6D pose, check with hugh

class AC_Network():
    def __init__(self,args, scope,trainer):
        self.scope = scope

        with tf.variable_scope(scope):
            #Input and visual encoding layers

            self.s_input = tf.placeholder(shape=[None,args.v_size,args.h_size,args.channels],dtype=tf.float32, name="s_input")
            self.t_input = tf.placeholder(shape=[None,args.v_size,args.h_size,args.channels],dtype=tf.float32, name="t_input")
            self.sensor_input = tf.placeholder(shape=[None, args.sensors],dtype=tf.float32, name="sensor_input")

            with tf.variable_scope("source"):
                self.source_net = GoogLeNet({'data': self.s_input}, trainable=False)

            with tf.variable_scope("target"):
                self.target_net = GoogLeNet({'data': self.t_input}, trainable=False)

            input_shape = self.source_net.get_output().get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                self.s_out = tf.reshape(self.source_net.get_output(), [-1, dim])
                self.t_out = tf.reshape(self.target_net.get_output(), [-1, dim])
            else:
                self.s_out = self.source_net.get_output()
                self.t_out = self.target_net.get_output()

            combined = tf.concat(values=[self.t_out, self.s_out,
                                               self.sensor_input], concat_dim=1)

            hidden = slim.fully_connected(combined, 256, activation_fn=tf.nn.elu)

            # Output layer for policy
            if args.discrete_actions:
                self.policy = slim.fully_connected(hidden,args.a_size,
                    activation_fn=tf.nn.softmax,
                    weights_initializer=normalized_columns_initializer(0.01),
                    biases_initializer=None, scope='policy')
            else:
                self.policy_means = slim.fully_connected(hidden,args.a_size,
                    activation_fn=None,
                    weights_initializer=normalized_columns_initializer(0.01),
                    biases_initializer=None, scope='policy_means')
                self.policy_variances = slim.fully_connected(hidden,args.a_size,
                    activation_fn=tf.nn.softplus,
                    weights_initializer=normalized_columns_initializer(0.01),
                    biases_initializer=None, scope='policy_variances')

            # Output layer for value estimation
            self.value = slim.fully_connected(hidden,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None, scope='value')
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                if args.discrete_actions:
                    self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                    self.actions_onehot = tf.one_hot(self.actions,args.a_size,dtype=tf.float32)
                    self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
                    self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                    self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                else:
                    self.actions = tf.placeholder(shape=[None,4],dtype=tf.float32)
                    self.entropy = - tf.reduce_sum(0.5 * (tf.log(2 * math.pi * self.policy_variances) + 1))
                    self.policy_dist = tf.contrib.distributions.MultivariateNormalDiag(self.policy_means,
                                                                                       tf.sqrt(self.policy_variances))
                    self.policy_loss = -tf.reduce_sum(tf.log(self.policy_dist.pdf(self.actions))*self.advantages)

                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

    def load(self, data_path, session, ignore_missing=False):
        with tf.variable_scope(self.scope):
            with tf.variable_scope("source"):
                self.source_net.load(data_path, session, ignore_missing)
            with tf.variable_scope("target"):
                self.target_net.load(data_path, session, ignore_missing)

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer
