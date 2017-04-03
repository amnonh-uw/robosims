import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
from PIL import Image
from images2gif import writeGif

#This code allows gifs to be saved of the training episodes
def make_gif(images, fname, duration=2):
  print(images.shape)
  image_list = []
  for i in range(0, images.shape[0]):
    img = Image.fromarray(images[i, :, :, :], 'RGB')
    image_list.append(img)

  writeGif(fname, image_list, duration)