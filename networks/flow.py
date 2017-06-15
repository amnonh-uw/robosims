import numpy as np
import os
import sys
import importlib
from PIL import Image, ImageDraw, ImageFont
from util.config import *
from util.util import *
from tensorflow.python import debug as tf_debug

def flow(argv):
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

    if args.image1 == None:
        print('must specify image1')
        exit()

    if args.image2 == None:
        print('must specify image2')
        exit()

    if args.output == None:
        print("must specify output")
        exit()

    im1 = load_image(args.image1)
    im1 = cls.preprocess_image(im1, keep_resolution=False)
    im1 = np.expand_dims(im1, axis=0)

    im2 = load_image(args.image2)
    im2 = cls.preprocess_image(im2, keep_resolution=False)
    im2 = np.expand_dims(im2, axis=0)


    im1_tensor = tf.placeholder(shape=im1.shape, dtype=tf.float32, name="im1")
    im2_tensor = tf.placeholder(shape=im2.shape, dtype=tf.float32, name="im2")

    phase = tf.constant(True, tf.bool, name='phase')
    net = cls({'data1': im1_tensor, 'data2': im2_tensor}, phase, fixed_resolution=False)

    accuracy = tf.constant(1)
    tf.summary.scalar('accuracy', accuracy)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter("log_test", sess.graph)
    output_tensor = net.get_output()

        # calc = [merged, output_tensor]
        # calc_name = ['merged', 'out']
        # for k in net.layers.keys():
        #     calc.append(net.layers[k])
        #     calc_name.append(k)

    print("loading network weights {}".format(cls_data))
    net.load(cls_data, sess, ignore_missing=True)
    print("network load complete")
        
    print("running...")
    sys.stdout.flush()

    out, merged  = sess.run([output_tensor, merged], {im1_tensor: im1, im2_tensor: im2})
    # vals = sess.run(calc, {im1_tensor: im1, im2_tensor: im2})
    # summary = vals[0]
    test_writer.add_summary(merged, 1)

    # out = vals[1]
    # for i in range(1, len(calc)):
    #    np.save('tf_blobs/' + calc_name[i] + '.npy', vals[i])

    out = out[0]
    writeFloFile(args.output, out)
    sess.close()

def load_image(path):
  """
  Loads image into 3D Numpy array of shape 
  (width, height, channels)
  """
  with Image.open(path) as image:         
    assert image.mode == "RGB"
    im = np.array(image, dtype=np.float32)

  return im

def writeFloFile(filename, data):
    with open(filename, 'wb') as f:
        f.write(b'PIEH')
        input_shape = data.shape
        h = np.int32(input_shape[0])
        w = np.int32(input_shape[1])
        print('writing an {}x{} flow file'.format(w, h))
        data = data.astype(np.float16)
        f.write(w)
        f.write(h)

        for y in range(0, h):
            for x in range(0, w):
                u = np.float32(data[y,x,0])
                v = np.float32(data[y,x,1])
                f.write(u)
                f.write(v)

if __name__ == "__main__":
    flow(sys.argv[1:])
