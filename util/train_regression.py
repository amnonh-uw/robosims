import tensorflow as tf
import math
import numpy as np
import os
import sys
import queue
import importlib
import signal
from robosims.unity import UnityGame
from util.util import *
from PIL import Image, ImageDraw, ImageFont

def train_regression(args, model_cls):
    conf = args.conf
    mod = importlib.import_module(conf.base_class)
    cls = getattr(mod, conf.base_class)
    cls_data = sys_path_find(conf.base_class + ".npy")
    if cls_data == None:
        print("can't find data file for class {}".format(conf.base_class))
        exit()

    cheat = False
    model = model_cls(conf, cls, cheat=cheat, trainable=True)
    make_dirs(model.name(), conf)

    # optimizer_update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # Create an optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)

    # Compute the gradients
    grads_and_vars = optimizer.compute_gradients(model.loss_tensor())

    # Ask the optimizer to apply the gradients.
    optimizer_update = optimizer.apply_gradients(grads_and_vars)

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(conf.log_path, sess.graph)
        summary_writer.flush()

        if args.load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(conf.model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            if conf.load_base_weights:
                print("loading network weights")
                model.network.load(cls_data, sess, ignore_missing=True)
                print("network load complete")
        
        env = UnityGame(args)
        try:
            episode_count = 0
            losses = queue.Queue()
            num_losses = 0
            mean_loss = 999999.
    
            if args.test_only:
                if args.iter == 0:
                    args.test_iter = 100
                else:
                    args.test_iter = args.iter
                train_iter = 0
            else:
                if args.iter == 0:
                    train_iter = 40000
                else:
                    train_iter = args.iter
                args.test_iter = 100
        
            print("doing {} training iterations".format(train_iter))
            for i in range(0, train_iter):
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
                        make_jpg(conf, "image_",  env, model, pred_value_out, episode_count, loss=loss)
                    if episode_count % 250 == 0:
                        saver.save(sess,conf.model_path+'/model-'+str(episode_count)+'.cptk')
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

            test(conf, sess, env, model, args.test_iter)

        except KeyboardInterrupt:
            print("W: interrupt received, stopping…")
            exit()
        finally:
            env.close()


def test(conf, sess, env, model, test_iter):
    print("testing... {} iterations".format(test_iter))
    for episode_count in range(0, test_iter):
        env.new_episode()
        t = env.get_state().target_buffer()
        s = env.get_state().source_buffer()

        t_input = process_frame(t)
        s_input = process_frame(s)

        feed_dict = { model.network.s_input:s_input, model.network.t_input:t_input}
        pred_value_out = sess.run([model.pred_tensor()], feed_dict=feed_dict)
        pred_value_out = pred_value_out[0]

        make_jpg(conf, "test_set_", env, model, pred_value_out, episode_count)

def process_frame(frame):
    need_shape = [1]
    need_shape += frame.shape
    return np.reshape(frame.astype(float)/ 255.0, need_shape)

def ascii_hist(x, bins):
    N,X = np.histogram(x, bins=bins)
    width = 50
    nmax = N.max()

    for (xi, n) in zip(X,N):
        bar = '#'*int(n*1.0*width/nmax)
        xi = '{0: <8.4g}'.format(xi).ljust(10)
        print('{0}| {1}'.format(xi,bar))

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
