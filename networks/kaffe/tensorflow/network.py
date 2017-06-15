import numpy as np
import tensorflow as tf
import math

DEFAULT_PADDING = 'SAME'


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, phase, trainable=False, fixed_resolution=False):
        # The input nodes for this network
        self.inputs = inputs
        # tensor telling us if we are training or testing (required for batch norm)
        self.phase = phase
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # if true, keep fully connected and softmax laters - network will not be resolution invariant
        self.fixed_resolution = fixed_resolution
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, encoding='bytes').item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iter(data_dict[op_name].items()):
                    try:
                        param_name_str = param_name.decode("utf-8")
                        param_name_str = param_name_str.replace('variance', 'bn/moving_variance')
                        param_name_str = param_name_str.replace('mean', 'bn/moving_mean')
                        param_name_str = param_name_str.replace('scale', 'bn/gamma')
                        param_name_str = param_name_str.replace('offset', 'bn/beta')
                        var = tf.get_variable(param_name_str)
                        var_name = var.name
                        var_shape = var.get_shape()
                        found = True
                    except ValueError:
                        if not ignore_missing:
                            print('cannot find variable {}'.format(param_name_str))
                            tvars = tf.trainable_variables()
                            for v in tvars:
                                print(v.name)
                            raise
                        else:
                            var_name = param_name_str
                            var_shape = '[unknown]'
                            found = False
                            print("missing {}/{}".format(op_name, param_name_str))
                    try:
                        if found:
                            session.run(var.assign(data))
                    except ValueError:
                        print('{} var shape {} vs data shape {}'.format(var_name, var_shape, data.shape))
                        print("value error on assign {}".format(var_name))

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)

            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')


    def make_deconv_filter(self, name, f_shape):
        width = f_shape[0]
        heigh = f_shape[0]
        f = math.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        var = tf.get_variable(name, shape=weights.shape, initializer=init, trainable=self.trainable)
        return var

    @layer
    def resize(self, input, new_height, new_width, name):
        with tf.variable_scope(name) as scope:
            return tf.image.resize_images(input, [new_height, new_width])

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             alpha=None,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True,
             weights=None):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape().as_list()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel

        input = tf.Print(input, [input], "convolution input for " + name)

        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            if weights is None:
                kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            else:
                weights = np.reshape(weights, [k_h, k_w, int(c_i / group), c_o])
                kernel = tf.constant(weights, dtype=tf.float32)

            kernel = tf.Print(kernel, [kernel], "kernel for " + name)

            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(input, 3, axis=group)
                kernel_groups = tf.split(kernel, 3, axis=group)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                biases = tf.Print(biases, [biases], "biases for " +  name)
                output = tf.nn.bias_add(output, biases)

            output = tf.Print(output, [output], "output before relu for " + name)
            if relu:
                # ReLU non-linearity
                if alpha is None:
                    output = tf.nn.relu(output)
                else:
                # Leaky ReLU non-linearity
                    m_output = tf.nn.relu(-output) * alpha
                    output = tf.nn.relu(output)
                    output -= m_output

            output = tf.Print(output, [output], "output after relu for " + name)
                    
            return output

    @layer
    def deconv(self,
               input,
               k_h,
               k_w,
               c_o,
               s_h,
               s_w,
               name,
               relu=True,
               alpha=None,
               padding=DEFAULT_PADDING,
               group=1,
               biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0

        with tf.variable_scope(name) as scope:
            # Compute output shape of input as tensor
            in_shape = tf.shape(input)
            h = in_shape[1] * s_h
            w = in_shape[2] * s_w
            new_shape = [in_shape[0], h, w, c_o]
            output_shape = tf.stack(new_shape)

            # compute output shape of input as list
            l_in_shape = input.get_shape().as_list()
            l_h = l_in_shape[1] * s_h
            l_w = l_in_shape[2] * s_w
            l0 = l_in_shape[0]
            if l0 == None:
                l0 = -1
            l_new_shape = [l0, l_h, l_w, c_o]

            # filter
            f_shape = [k_h, k_w, c_o, c_i]
            weights = self.make_deconv_filter('weights', f_shape)
            output = tf.nn.conv2d_transpose(input, weights, output_shape, [1, s_h, s_w, 1], padding=padding,  name=scope.name)

            output = tf.reshape(output, l_new_shape)
        
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)

            if relu:
                # ReLU non-linearity
                if alpha is None:
                    output = tf.nn.relu(output, name=scope.name)
                else:
                # Leaky ReLU non-linearity
                    m_output = tf.nn.relu(-output)
                    output = tf.nn.relu(output)
                    output = tf.subtract(output, alpha * m_output, name=scope.name)

        return output

    @layer
    def subtract_mean(self, input, input_scale, means, name):
        with tf.variable_scope(name) as scope:
            return input_scale * input - means

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def prelu(self, input, name):
        with tf.variable_scope(name):
            i = input.get_shape().as_list()
            i = i[1:]
            init = np.zeros(i, dtype=np.float32)
            init.fill(0.25)
            alphas = tf.get_variable('alpha', dtype=tf.float32, trainable=self.trainable, initializer=tf.constant(init))

            output = tf.nn.relu(input) + alphas * (input - tf.abs(input)) * 0.5

        return output

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)


    @layer
    def add(self, inputs, name, coeff=None):
        if isinstance(inputs, list):
            if coeff != None:
                assert "coeff not None and input is list not implemnted"
            else:
                return tf.add_n(inputs, name=name)
        else:
            assert coeff != None
            return tf.multiply(inputs, coeff, name=name)


    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = list(map(lambda v: v.value, input.get_shape()))
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name=name)

    @layer
    def batch_normalization(self, input, name, relu=False, decay=0.999, epsilon=1e-5):
        with tf.variable_scope(name) as scope:
            output = tf.contrib.layers.batch_norm(input,
                                          center=True, scale=True, 
                                          is_training=self.phase,
                                          decay=decay,
                                          epsilon=epsilon,
                                          scope='bn')
            if relu:
                output = tf.nn.relu(output)

        return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)
