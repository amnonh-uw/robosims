import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import numpy as np
import os
import sys
import imageio
import signal
from robosims.unity import UnityGame
from PIL import Image, ImageDraw, ImageFont

def make_dirs(default_postfix, args):
    conf = args.conf
    postfix = default_postfix

    if args.postfix is not None:
        postfix = args.postfix
    elif conf.postfix != None:
        postfix = conf.postfix

   # Create a directory to save the model to
    conf.model_path = './model_' + postfix
    make_dir(conf.model_path, conf)

    # Create a directory to save episode playback gifs to
    conf.frames_path = './frames_' + postfix
    make_dir(conf.frames_path, conf)

    # Create a directory for logging
    conf.log_path = './log_' + postfix
    make_dir(conf.log_path, conf)

def make_dir(file_path, conf):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    conf_path = file_path + '/conf.txt'
    with open(conf_path, 'w') as f:
        f.write(str(conf))
        f.write('\n')

def make_gif(conf, episode_target_frame, episode_source_frames, episode_count):
    gif_file = conf.frames_path + '/movie_'+str(episode_count)+'.gif'
    writer = imageio.get_writer(gif_file, mode='I', fps=conf.gif_fps)

    im_t = Image.fromarray(episode_target_frame)

    for frame in episode_source_frames:
        im_s = Image.fromarray(frame)
        im = make_image((im_s, im_t), ("source", "dest"))
        writer.append_data(PIL2array(im))

    writer.close()

def make_image(images, cap_texts, cap_colors):
    seperator_width = 4
    text_height_margin = 4
    text_width_margin = 10
    num_caps = len(cap_texts[0])

    image0 = images[0]

    font = get_font(16)
    image0_draw = ImageDraw.Draw(image0)
    _, text_height = image0_draw.textsize("Hello", font=font)

    caption_height = (text_height + text_height_margin) * num_caps + 2 * text_height_margin
    seperator_height = image0.size[1]
    seperator = Image.new('RGB', (seperator_width, seperator_height))

    # calculae width, build bottom list
    width = -seperator_width
    bottom_list = list()
    for i in range(len(images)):
        width += images[i].size[0] + seperator_width
        bottom_list.append(images[i])
        if i != len(images) - 1:
            bottom_list.append(seperator)

    # build caption
    caption = Image.new('RGB', (width, caption_height))
    caption_draw = ImageDraw.Draw(caption)

    width_start = text_width_margin
    for i in range(len(images)):
        caps = cap_texts[i]
        colors = cap_colors[i]
        line_height = text_height_margin

        for j in range(len(caps)):
            caption_draw.text((width_start,line_height), caps[j], font=font, fill=colors[j])
            line_height += text_height_margin + text_height

        width_start += seperator_width + images[i].size[0]

    bottom = hstack_images(bottom_list)
    stacked = vstack_images((caption, bottom))

    return stacked

def make_jpg(conf, prefix, array_images, cap_texts, cap_colors, episode_count):
    images = []
    for im in array_images:
        images.append(Image.fromarray(im))

    stacked = make_image(images, cap_texts, cap_colors)
    stacked.save(conf.frames_path + "/" +  prefix + str(episode_count)+'.jpg')

def vstack_images(images):
    total_width = images[0].size[0]
    total_height = 0
    for im in images:
        total_height += im.size[1]

    new_im = Image.new('RGB', (total_width, total_height))

    h_offset = 0
    for im in images:
        new_im.paste(im, (0, h_offset))
        h_offset += im.size[1]

    return new_im

def hstack_images(images):
    total_width = 0
    total_height = images[0].size[1]
    for im in images:
        total_width += im.size[0]

    new_im = Image.new('RGB', (total_width, total_height))

    w_offset = 0
    for im in images:
        new_im.paste(im, (w_offset, 0))
        w_offset += im.size[0]

    return new_im

def flatten(t, max_size=0):
    if isinstance(t, list):
        l = []
        for t_item in t:
            l.append(flatten(t_item, max_size))

        return l

    input_shape = t.get_shape()
    if input_shape.ndims == 4:
        dim = 1
        for d in input_shape[1:].as_list():
            dim *= d
        print("flattening " + t.name + " size " + str(dim))
        t = tf.reshape(t, [-1, dim])
    elif input_shape.ndims == 1 or input_shape.ndims == 2:
        dim = input_shape[1]
    else:
        raise ValueError("invalid number of dimensions " + str(input_shape.ndims))

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

def sys_path_find(pathname, matchFunc=os.path.isfile):
    for dirname in sys.path:
        candidate = os.path.join(dirname, pathname)
        if matchFunc(candidate):
            return candidate

    return None

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    tf.summary.scalar(var.name, var)

def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def optimistic_restore(session, save_file):
    print('optimisitc restore')
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    print(saved_shapes)
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                print('will resotre {}'.format(curr_var))
                restore_vars.append(curr_var)
            else:
                print('shape for {} doesnt match'.format(curr_var))
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def map_error(a, f, eps=0.001):
    n = 1
    if isinstance(a, np.ndarray):
        a = np.ravel(a)
        n = a.shape[0]
    if isinstance(f, np.ndarray):
        f = np.ravel(f)

    return np.sum(abs((a - f)/(a+eps))) / n

def abs_error(a, f):
    n = 1
    if isinstance(a, np.ndarray):
        a = np.ravel(a)
        n = a.shape[0]
    if isinstance(f, np.ndarray):
        f = np.ravel(f)

    return np.sum(abs((a - f))) / n

def map_accuracy(a, f, eps=0.001):
    return 1 - map_error(a, f, eps)

def abs_accuracy(a, f):
    return 1 - abs_error(a, f)

def  as_vector(a, dim):
    if a.size != dim:
        raise ValueError("as_vector excpects pred_value to be of size " + str(dim))

    if a.shape[0] != dim:
        a = a[0]
        if a.shape[0] != dim:
            a = a[0]
            if a.shape[0] != dim:
                raise ValueError("as_vector excpects pred_value to be of size " + str(dim))

    return a

def as_scalar(a):
    return as_vector(a, 1)

def process_frame(frame, cls):
    f = frame.astype(float)
    f = cls.preprocess_image(f, fixed_resolution=False)
    return f

def dataset_files(dataset):
    idx_file = dataset + ".idx"
    data_file = dataset + ".data"

    return data_file, idx_file

def get_font(fontsize):
    fontfile = "fonts/Menlo-Regular.ttf"
    try:
        font = ImageFont.truetype(fontfile, fontsize)
    except OSError:
        print("cannot load {}, using default font".format(fontfile))
        font = ImageFont.load_default()

    return font

