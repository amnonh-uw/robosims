import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import queue
import importlib
import signal
from robosims.unity import UnityGame
from learn_direction.direction_network import *

def train(argv):
    parser = argparse.ArgumentParser(description='a3c train')
    parser.add_argument('--h_size', type=int, help='horizontal image size', default=400)
    parser.add_argument('--v_size', type=int, help='vertical image size', default=300)
    parser.add_argument('--channels', type=int, help='image channels', default=3)
    parser.add_argument('--sensors', type=int, help='sensor channels', default=1)
    parser.add_argument('--gamma', type=float, help='discount rate', default=0.99)
    parser.add_argument('--config', type=str, help='config file', default="")
    parser.add_argument('--load-model', dest='load_model', action='store_true')
    parser.add_argument('--no-load_model', dest='load_model', action='store_false')
    parser.add_argument('--discrete_action_distance', type=float, default=0.01)
    parser.add_argument('--discrete_action_rotation', type=float, default=1)
    parser.add_argument('--close-enough-distance', type=float, default = 0.01)
    parser.add_argument('--close-enough-rotation', type=float, default = 1)
    parser.add_argument('--max-distance-delta', type=float, default = 1)
    parser.add_argument('--max-rotation-delta', type=float, default = 3)
    parser.add_argument('--collision-reward', type=int, default = -1000)
    parser.add_argument('--step-reward', type=int, default = -1)
    parser.add_argument('--close-enough-reward', type=int, default = 1000)
    parser.add_argument('--discrete-actions', dest='discrete_actions', action='store_true')
    parser.add_argument('--continous_actions', dest='discrete_actions', action='store_false')
    parser.add_argument('--base-class', type=str, default='GoogleNet')
    parser.add_argument('--load-base-weights', dest='load_base_weights', action='store_true')
    parser.set_defaults(load_model=False)
    parser.set_defaults(load_base_weights=False)
    parser.set_defaults(discrete_actions=False)

    args = parser.parse_args(argv)
    print(args)

    mod = importlib.import_module(args.base_class)
    cls = getattr(mod, args.base_class)
    cls_data = sys_path_find(args.base_class + ".npy")
    if cls_data == None:
        print("can't find data file for class {}".format(args.base_class))
        exit()

    learning_rate = 0.01

    # Create a directory to save the model to
    model_path = './model_direction'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Create a directory to save episode playback gifs to
    frames_path = './frames_direction'
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    # Create a directory for logging
    log_path = './log_direction'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Construct model
    cheat = None
    # cheat = tf.placeholder(tf.float32, shape=[None, 3], name='cheat')
    network = Direction_Network(args, cls, "main", trainable=False)

    direction_pred = network.get_output()
    direction = tf.placeholder(tf.float32, shape=[None, 3], name='direction')

    # l2 loss of real direction - predicted
    loss = tf.nn.l2_loss(direction - direction_pred)
    chance_loss = 3 * 0.5 * args.max_distance_delta * args.max_distance_delta

    # optimizer_update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # Create an optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Compute the gradients
    grads_and_vars = optimizer.compute_gradients(loss)

    # Ask the optimizer to apply the gradients.
    optimizer_update = optimizer.apply_gradients(grads_and_vars)

    saver = tf.train.Saver(max_to_keep=5)

    weight_mean = None
    for v in tf.trainable_variables():
        print(v.name)
        if weight_mean is None:
            weight_mean = tf.reduce_mean(v)
        else:
            weight_mean += tf.reduce_mean(v)

    if weight_mean is None:
        print("didn't find weight to track")
        exit()

    with tf.Session() as sess:
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(log_path, sess.graph)
        summary_writer.flush()

        if args.load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            if args.load_base_weights:
                print("loading network weights")
                network.load(cls_data, sess, ignore_missing=True)
                print("network load complete")
        
        env = UnityGame(args)
        try:
            episode_count = 0
            losses = queue.Queue()
            test_set = queue.Queue()
            num_losses = 0
            mean_loss = 999999.
            weight_mean_old = 0
    
            while True:
                env.new_episode()
                episode_count += 1
                t = env.get_state().target_buffer()
                s = env.get_state().source_buffer()

                t_input = process_frame(t)
                s_input = process_frame(s)
                true_direction = np.reshape(env.direction(), [1,3])

                if cheat is not None:
                    s_input = np.zeros(s_input.shape)
                    t_input = np.zeros(s_input.shape)
                    cheat_direction = 2 * true_direction
                    feed_dict = {
                             network.s_input:s_input,
                             network.t_input:t_input,
                             cheat:cheat_direction,
                             direction:true_direction}
                
                    check_grad(sess, grads_and_vars, feed_dict)

                    _, l, weight_mean_out = sess.run([
                            optimizer_update,
                            loss,
                            weight_mean], feed_dict=feed_dict)

                else:
                    feed_dict = {network.s_input:s_input,
                             network.t_input:t_input,
                             direction:true_direction}

                    check_grad(sess, grads_and_vars, feed_dict)

                    _, l, weight_mean_out = sess.run([
                            optimizer_update,
                            loss,
                            weight_mean], feed_dict=feed_dict)


                losses.put(l)
                test_set.put((s, t))
                if num_losses <= 100:
                    num_losses += 1
                else:
                    losses.get()
                    test_set.get()

                m = np.mean(np.asarray(losses.queue))
                if weight_mean_old == 0:
                    weight_mean_delta = 0
                else:
                    weight_mean_delta = abs(weight_mean_old - weight_mean_out)
                    weight_mean_old = weight_mean_out
                print("Loss={} weight_mean_delta={} avg_loss={}". format(l, weight_mean_delta, m))

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 25 == 0:
                        time_per_step = 0.05
                        make_jpg(frames_path, "image_", s, t, episode_count, l, chance_loss)
                    if episode_count % 250 == 0:
                        saver.save(sess,model_path+'/model-'+str(episode_count)+'.cptk')
                        print("Saved Model")

                if episode_count % 300 == 0:
                    print("loss running average {} vs chance loss {}".format(m, chance_loss))
                    print("loss histogram")
                    ascii_hist(np.asarray(losses.queue), bins=50)
                    if m >= mean_loss:
                        print("mean loss stable at {}".format(m))
                        break
                    else:
                        mean_loss = m

        except KeyboardInterrupt:
            print("W: interrupt received, stopping…")
            exit()
        finally:
            env.close()

    episode_count = 1
    while not losses.empty():
        l = losses.get()
        s, t = test_set.get()
        make_jpg(frames_path, "test_set_", s, t, episode_count, l, chance_loss)
        episode_count += 1

def process_frame(frame):
    need_shape = [1]
    need_shape += frame.shape
    return np.reshape(frame.astype(float)/ 255.0, need_shape)

from PIL import Image, ImageDraw, ImageFont
def make_jpg(frames_path, prefix, s, t, episode_count, l, chance_loss):
    both = np.vstack((s, t))
    im_both = Image.fromarray(both, 'RGB')

    # get a font
    fnt = ImageFont.load_default()
    # get a drawing context
    d = ImageDraw.Draw(im_both)

    # draw text
    d.text((10,10), "loss="+ str(l) + " vs chance " + str(chance_loss), font=fnt)
    im_both.save(frames_path + "/" +  prefix + str(episode_count)+'.jpg')

def ascii_hist(x, bins):
    N,X = np.histogram(x, bins=bins)
    width = 50
    nmax = N.max()

    for (xi, n) in zip(X,N):
        bar = '#'*int(n*1.0*width/nmax)
        xi = '{0: <8.4g}'.format(xi).ljust(10)
        print('{0}| {1}'.format(xi,bar))

def sys_path_find(pathname, matchFunc=os.path.isfile):
    for dirname in sys.path:
        candidate = os.path.join(dirname, pathname)
        if matchFunc(candidate):
            return candidate
    
    return None

def check_grad(sess, grads_and_vars, feed_dict):
    for gv in grads_and_vars:
        grad = sess.run(gv[1], feed_dict=feed_dict)
        if zero_grad(grad):
            print("Zero gradient!")
            print(str(grad) + " - " + gv[1].name)
            print(str(sess.run(gv[0],
                feed_dict=feed_dict)) + " - gradient " + gv[1].name)

def zero_grad(grad):
    if np.count_nonzero(grad) != 0:
            return False

    return True

if __name__ == "__main__":
    train(sys.argv[1:])
