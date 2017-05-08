import math
import numpy as np
import os
import sys
import queue
import importlib
import signal
from robosims.unity import UnityGame
from util.util import *
from util.laplotter import *
from PIL import Image, ImageDraw, ImageFont

def train_regression(args, model_cls):
    conf = args.conf
    mod = importlib.import_module(conf.base_class)
    cls = getattr(mod, conf.base_class)
    if conf.load_base_weights:
        cls_data = sys_path_find(conf.base_class + ".npy")
        if cls_data == None:
            print("can't find data file for class {}".format(conf.base_class))
            exit()

    cheat = conf.cheat
    model = model_cls(conf, cls, cheat=cheat, trainable=True)
    make_dirs(model.name(), args)

    # optimizer_update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # Create an optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)

    # Compute the gradients
    grads_and_vars = optimizer.compute_gradients(model.loss_tensor())

    # Ask the optimizer to apply the gradients.
    optimizer_update = optimizer.apply_gradients(grads_and_vars)

    saver = tf.train.Saver(max_to_keep=5)

    plotter = LossAccPlotter(save_to_filepath=conf.frames_path + "/chart.png")

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
        
        try:
            episode_count = 0
            losses = queue.Queue()
            num_losses = 0
    
            if args.test_only:
                if args.iter == 0:
                    args.test_iter = 100
                else:
                    args.test_iter = args.iter
                train_iter = 0
                train_outer_iter = 0
            else:
                if args.iter == 0:
                    train_iter = 40000
                else:
                    train_iter = args.iter
                if args.outer_iter == 0:
                    train_outer_iter = 1
                else:
                    train_outer_iter = args.outer_iter
                args.test_iter = 100
        
            print("doing {}*{} training iterations".format(train_iter, train_outer_iter))
            for i in range(0, train_outer_iter):
                env = UnityGame(args)
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
                
                        if conf.check_gradients:
                            check_grad(sess, grads_and_vars, feed_dict)

                        _, loss, pred_value_out, summary = sess.run([
                            optimizer_update,
                            model.loss_tensor(),
                            model.pred_tensor(),
                            model.summary_tensor()], feed_dict=feed_dict)

                    else:
                        feed_dict = {
                            model.network.s_input:s_input,
                            model.network.t_input:t_input,
                            model.true_tensor():model.true_value(env)}

                        if conf.check_gradients:
                            check_grad(sess, grads_and_vars, feed_dict)

                        _, loss, pred_value_out, summary = sess.run([
                            optimizer_update,
                            model.loss_tensor(),
                            model.pred_tensor(),
                            model.summary_tensor()], feed_dict=feed_dict)

                    # print("pred_value_out is {}".format(pred_value_out))
                    # print("loss is {}".format(loss))
                    loss = np.asscalar(loss)
                    summary_writer.add_summary(summary, i)
                    losses.put(loss)
                    if num_losses <= 100:
                        num_losses += 1
                    else:
                        losses.get()

                    m = np.mean(np.asarray(losses.queue))

                    acc_train = model.accuracy(env, pred_value_out)
                    loss_train = loss
                    if abs(acc_train) > 2:
                        acc_train = None
                    if abs(loss_train) > 2:
                        loss_train = None

                    plotter.add_values(episode_count, loss_train=loss_train, acc_train=acc_train, redraw=False)

                    # Periodically save gifs of episodes, model parameters, and summary statistics.
                    if episode_count != 0:
                        if episode_count % conf.flush_plot_frequency == 0:
                            make_train_jpg(conf, "image_",  env, model, pred_value_out, episode_count, loss=loss)
                            print("{}: Loss={} avg_loss={}".format(episode_count, loss, m))
                            summary_writer.flush()
                            plotter.redraw()

                        if episode_count % conf.model_save_frequency == 0:
                            saver.save(sess,conf.model_path+'/model-'+str(episode_count)+'.cptk')
                            print("Saved Model")

                env.close()
            test(conf, sess, env, model, args.test_iter)

        except KeyboardInterrupt:
            print("W: interrupt received, stopping…")
            exit()
        except (EOFError):
            print("end of pickled file reached")
        finally:
            env.close()

    plotter.redraw()

def predict(sess, env, model):
        t = env.get_state().target_buffer()
        s = env.get_state().source_buffer()

        t_input = process_frame(t)
        s_input = process_frame(s)

        feed_dict = { model.network.s_input:s_input, model.network.t_input:t_input}
        pred_value_out = sess.run([model.pred_tensor()], feed_dict=feed_dict)
        return (pred_value_out[0])[0]

def make_train_jpg(conf, prefix,  env, model, pred_value_out, episode_count, loss):
    pred_value = pred_value_out[0]
    t = env.get_state().target_buffer()
    s = env.get_state().source_buffer()
    images = [t, s]
    err_str = model.error_str(env, pred_value)
    cap_texts = ["target:" + env.target_str(), "source:" + env.source_str()]
    cap_texts2 = [ err_str, "loss{} ".format(loss) ]

    make_jpg(conf, prefix, images, cap_texts, cap_texts2,  episode_count)

def test(conf, sess, env, model, test_iter):
    print("testing... {} iterations".format(test_iter))
    for episode_count in range(0, test_iter):
        env.new_episode()
        pred_value = predict(sess, env, model)

        t = env.get_state().target_buffer()
        s = env.get_state().source_buffer()
        images = [t, s]
        err_str = model.error_str(env, pred_value)
        cap_texts = ["target:" + env.target_str(), "source:" + env.source_str()]
        cap_texts2 = [ err_str, "" ]

        if conf.test_steps == 0:
            make_jpg(conf, "test_set_", images, cap_texts, cap_texts2,  episode_count)
        else:
            for step in range(conf.test_steps):
                x = float(pred_value[0])
                y = float(pred_value[1])
                z = float(pred_value[2])
                env.take_prediction_step(x, y,z)
                image = env.get_state().source_buffer()
                images.append(image)
                pred_value = predict(sess, env, model)
                err_str = model.error_str(env, pred_value)
                cap_texts.append("step {}:{}".format(step+1, env.source_str()))
                cap_texts2.append(err_str)

            make_jpg(conf, "test_set_steps_", images, cap_texts, cap_texts2, episode_count)

def process_frame(frame):
    need_shape = [1]
    need_shape += frame.shape
    return np.reshape(frame.astype(float)/ 255.0, need_shape)

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
