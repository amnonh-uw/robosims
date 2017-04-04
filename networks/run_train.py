import tensorflow as tf
import math
import numpy as np
import argparse
import os
import sys
import queue
import importlib
import signal
from robosims.unity import UnityGame
from PIL import Image, ImageDraw, ImageFont

def run_train(args, model_cls):
    mod = importlib.import_module(args.base_class)
    cls = getattr(mod, args.base_class)
    cls_data = sys_path_find(args.base_class + ".npy")
    if cls_data == None:
        print("can't find data file for class {}".format(args.base_class))
        exit()

    learning_rate = 1e-4

    cheat = False
    model = model_cls(args, cls, cheat=cheat, trainable=True)
    make_dirs(model.name(), args)

    # optimizer_update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # Create an optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Compute the gradients
    grads_and_vars = optimizer.compute_gradients(model.loss_tensor())

    # Ask the optimizer to apply the gradients.
    optimizer_update = optimizer.apply_gradients(grads_and_vars)

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(args.log_path, sess.graph)
        summary_writer.flush()

        if args.load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(args.model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            if args.load_base_weights:
                print("loading network weights")
                model.network.load(cls_data, sess, ignore_missing=True)
                print("network load complete")
        
        env = UnityGame(args)
        try:
            episode_count = 0
            losses = queue.Queue()
            num_losses = 0
            mean_loss = 999999.
    
            for i in range(0, 40000):
                env.new_episode()
                episode_count += 1
                t = env.get_state().target_buffer()
                s = env.get_state().source_buffer()

                t_input = process_frame(t)
                s_input = process_frame(s)

                if cheat:
                    s_input = np.zeros(s_input.shape)
                    t_input = np.zeros(s_input.shape)

                    feed_dict = {
                             model.network.s_input:s_input,
                             model.network.t_input:t_input,
                             model.cheat_tensor():model.cheat_value(env),
                             model.true_tensor():model.true_value(env)}
                
                    check_grad(sess, grads_and_vars, feed_dict)

                    _, l, pred_value_out = sess.run([
                            optimizer_update,
                            model.loss_tensor(),
                            model.true_tensor()], feed_dict=feed_dict)

                else:
                    feed_dict = {model.network.s_input:s_input,
                                 model.network.t_input:t_input,
                                 model.true_tensor():model.true_value(env)}

                    check_grad(sess, grads_and_vars, feed_dict)

                    _, loss, pred_value_out = sess.run([
                            optimizer_update,
                            model.loss_tensor(),
                            model.pred_tensor()], feed_dict=feed_dict)

                losses.put(loss)
                if num_losses <= 100:
                    num_losses += 1
                else:
                    losses.get()

                m = np.mean(np.asarray(losses.queue))

                print("{}: Loss={} avg_loss={}".format(episode_count, loss, m))

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if episode_count % 25 == 0:
                        time_per_step = 0.05
                        make_jpg(args.frames_path, "image_",  env, model, pred_value_out, episode_count, loss=loss)
                    if episode_count % 250 == 0:
                        saver.save(sess,args.model_path+'/model-'+str(episode_count)+'.cptk')
                        print("Saved Model")

                if episode_count % 200 == 0:
                    print("loss running average {} vs chance loss {}".format(m, model.chance_loss()))
                    print("loss histogram")
                    ascii_hist(np.asarray(losses.queue), bins=50)
                    if m >= mean_loss:
                        print("mean loss stable at {}".format(m))
                        # break
                    else:
                        mean_loss = m

            test(sess, env, model, args.frames_path)

        except KeyboardInterrupt:
            print("W: interrupt received, stopping…")
            exit()
        finally:
            env.close()


def test(sess, env, model, frames_path):
    print("testing...")
    for episode_count in range(1, 100):
        env.new_episode()
        t = env.get_state().target_buffer()
        s = env.get_state().source_buffer()

        t_input = process_frame(t)
        s_input = process_frame(s)

        feed_dict = { model.network.s_input:s_input, model.network.t_input:t_input}
        pred_value_out = sess.run([model.pred_tensor()], feed_dict=feed_dict)
        pred_value_out = pred_value_out[0]

        make_jpg(frames_path, "test_set_", env, model, pred_value_out, episode_count)

def process_frame(frame):
    need_shape = [1]
    need_shape += frame.shape
    return np.reshape(frame.astype(float)/ 255.0, need_shape)


def make_jpg(frames_path, prefix, env, model, pred_value, episode_count, loss=None):
    t = env.get_state().target_buffer()
    s = env.get_state().target_buffer()
    both = np.vstack((s, t))
    im_both = Image.fromarray(both, 'RGB')

    # get a font
    fnt = ImageFont.load_default()
    # get a drawing context
    d = ImageDraw.Draw(im_both)

    a = model.error_str(env, pred_value)

    if loss is None:
        draw_text = a
    else:
        draw_text = "loss="+ str(loss) + " " + a

    # draw text
    d.text((10,10), draw_text, font=fnt)
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

def parse_args(argv):
    print(argv)
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
    return args


def make_dirs(postfix, args):
   # Create a directory to save the model to
    args.model_path = './model_' + postfix
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Create a directory to save episode playback gifs to
    args.frames_path = './frames_' + postfix
    if not os.path.exists(args.frames_path):
        os.makedirs(args.frames_path)

    # Create a directory for logging
    args.log_path = './log_' + postfix
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
