import numpy as np
import os
import sys
import importlib
from PIL import Image, ImageDraw, ImageFont
from util.config import *
from util.util import *

def classify(argv):
    args = parse_args(argv)
    if args.base_class == None:
        print('must specify base-class')
        exit()

    base_class = args.base_class
    mod = importlib.import_module(base_class)
    cls = getattr(mod, base_class)

    cls_data = sys_path_find(base_class + ".npy")
    if cls_data == None:
        print("can't find data file for class {}".format(base_class))
        exit()

    if args.image == None:
        print('must specify image')
        exit()


    im = load_image(args.image)
    im = cls.preprocess_image(im, fixed_resolution=True)
    im = np.expand_dims(im, axis=0)
    im_tensor = tf.placeholder(shape=im.shape,dtype=tf.float32, name="input")

    phase = tf.constant(True, tf.bool, name='phase')
    net = cls({'data': im_tensor}, phase, trainable=False, fixed_resolution=True)
    cls_tensor = net.get_output()
    synset = [l.strip() for l in open('images/synset.txt').readlines()]

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        print("loading network weights")
        net.load(cls_data, sess, ignore_missing=False)
        print("network load complete")
        
        prob  = sess.run(cls_tensor, {im_tensor: im})
        top5 = np.argsort(prob[0])[-1:-6:-1]

        for n in top5:
            print("{0:.2f}: {1}".format(prob[0,n], synset[n]))

def load_image(path):
  """
  Loads JPEG image into 3D Numpy array of shape 
  (width, height, channels)
  """
  with Image.open(path) as image:         
    assert image.mode == "RGB"
    im = np.array(image)

  return im

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    classify(sys.argv[1:])
