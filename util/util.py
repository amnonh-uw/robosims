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

def make_dirs(postfix, conf):
   # Create a directory to save the model to
    conf.model_path = './model_' + postfix
    if not os.path.exists(conf.model_path):
        os.makedirs(conf.model_path)

    # Create a directory to save episode playback gifs to
    conf.frames_path = './frames_' + postfix
    if not os.path.exists(conf.frames_path):
        os.makedirs(conf.frames_path)

    # Create a directory for logging
    conf.log_path = './log_' + postfix
    if not os.path.exists(conf.log_path):
        os.makedirs(conf.log_path)

def make_gif(conf, episode_target_frame, episode_source_frames, episode_count):
    gif_file = conf.frames_path + '/movie_'+str(episode_count)+'.gif'
    writer = imageio.get_writer(gif_file, mode='I', fps=conf.gif_fps)

    im_t = Image.fromarray(episode_target_frame)

    writer.append_data(episode_target_frame)
    for frame in episode_source_frames:
        im_s = Image.fromarray(frame)
        im = make_image(im_s, im_t, "source", "dest")
        writer.append_data(PIL2array(im))

    writer.close()

def make_image(source, dest, source_caption_text, dest_caption_text = ""):
    seperator_width = 4
    text_height_margin = 2
    source_width_margin = 10

    font = ImageFont.load_default()
    source_draw = ImageDraw.Draw(source)
    _, text_height = source_draw.textsize(source_caption_text, font=font)
    text_height += 2 * text_height_margin
    dest_width_margin = source.size[0] + seperator_width + source_width_margin
    width = source.size[0] + dest.size[0] + seperator_width

    caption = Image.new('RGB', (width, text_height))
    caption_draw = ImageDraw.Draw(caption)
    caption_draw.text((source_width_margin,text_height_margin), source_caption_text, font=font)
    caption_draw.text((dest_width_margin,text_height_margin), dest_caption_text, font=font)

    height = source.size[1] + text_height
    seperator = Image.new('RGB', (seperator_width, height))

    bottom = hstack_images((source, seperator, dest))
    stacked = vstack_images((caption, bottom))

    return stacked

def make_jpg(conf, prefix, env, model, pred_value, episode_count, loss=None):
    t = env.get_state().target_buffer()
    im_t = Image.fromarray(t)
    s = env.get_state().source_buffer()
    im_s = Image.fromarray(s)
    err_str = model.error_str(env, pred_value)

    if loss is None:
        draw_text = err_str
    else:
        draw_text = "loss="+ str(loss) + " " + err_str

    stacked = make_image(im_s, im_t, "source " + draw_text, "dest")
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
    elif input_shape.ndims == 1:
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

def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)