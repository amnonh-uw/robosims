import tensorflow as tf
import math
import numpy as np
import os
import sys
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


def make_jpg(conf, prefix, env, model, pred_value, episode_count, loss=None):
    t = env.get_state().target_buffer()
    im_t = Image.fromarray(t)
    s = env.get_state().target_buffer()
    im_s = Image.fromarray(s)
    err_str = model.error_str(env, pred_value)

    if loss is None:
        draw_text = err_str
    else:
        draw_text = "loss="+ str(loss) + " " + err_str

    fnt = ImageFont.load_default()
    im_t_draw = ImageDraw.Draw(im_t)
    _, h = im_t_draw.textsize(draw_text, font=fnt)
    w = s.shape[0]
    h += 4

    caption = Image.new('RGB', (w, h))
    caption_draw = ImageDraw.Draw(caption)
    caption_draw.text((10,2), draw_text, font=fnt)

    seperator = Image.new('RGB', (w, 4))

    stacked = vstack_images((caption, im_s, seperator, im_t))
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
