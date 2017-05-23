import math
import numpy as np
import os
import sys
import queue
import importlib
import signal
import traceback
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
    model = model_cls(conf, cls, pose_dims=conf.pose_dims, cheat=cheat, trainable=True)
    make_dirs(model.name(), args)

    # Create an optimizer.
    if conf.use_adam:
        optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=conf.learning_rate)
        
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Ensures that we execute the update_ops before calculating
    # gradients and aplying them
    #with tf.control_dependencies(update_ops):
        # Compute the gradients
    grads_and_vars = optimizer.compute_gradients(model.loss_tensor(), colocate_gradients_with_ops=conf.colocate_gradients_with_ops)

        # Ask the optimizer to apply the gradients.
    optimizer_update = optimizer.apply_gradients(grads_and_vars)

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=conf.allow_soft_placement,
                                          log_device_placement=conf.log_device_placement)) as sess:
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(conf.log_path, sess.graph)
        summary_writer.flush()

        if args.load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(conf.model_path)
            if ckpt == None:
                print("No saved checkpoints found")
                exit()

            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            if conf.load_base_weights:
                print("loading network weights")
                model.network.load(cls_data, sess, ignore_missing=True)
                print("network load complete")
        
        try:
            episode_count = 0
            batch_count = 0
    
            if args.test_only:
                train_iter = 0
                epochs = 0
            else:
                plotter = LossAccPlotter(save_to_filepath=conf.frames_path + "/chart.png")
                plotter.averages_period = conf.averages_period

                train_iter = conf.iter
                epochs = conf.epochs

                print("doing {}*{} training iterations".format(train_iter, epochs))
        
            env = None
            episodes_in_batch = 0
            for epoch in range(0, epochs):
                print("epoch {}".format(epoch))
                env = UnityGame(conf, num_iter=train_iter)
                for i in range(0, train_iter):
                    env.new_episode()
                    episode_count += 1

                    if episodes_in_batch == 0:
                        t_input = list()
                        s_input = list()
                        true_values = list()
                        if cheat:
                            cheat_values = list()

                    t = env.get_state().target_buffer()
                    s = env.get_state().source_buffer()

                    t_input.append(process_frame(t, cls))
                    s_input.append(process_frame(s, cls))
                    true_values.append(model.true_value(env))

                    if cheat:
                        cheat_values.append(model.cheat_value(env))

                    episodes_in_batch += 1
                    if episodes_in_batch != conf.batch_size:
                        continue
                    episodes_in_batch = 0
                    batch_count += 1

                    s_input = np.array(s_input)
                    t_input = np.array(t_input)
                    true_values = np.array(true_values)

                    if cheat:
                        cheat_values = np.array(cheat_values)

                        feed_dict = {
                             model.phase_tensor():0,
                             model.network.s_input:s_input,
                             model.network.t_input:t_input,
                             model.cheat_tensor():cheat_values,
                             model.true_tensor():true_values}
                    else:
                        feed_dict = {
                            model.phase_tensor():0,
                            model.network.s_input:s_input,
                            model.network.t_input:t_input,
                            model.true_tensor():true_values}
                
                    if conf.check_gradients:
                        check_grad(sess, grads_and_vars, feed_dict)

                    _, loss, pred_values, errors, summary = sess.run([
                            optimizer_update,
                            model.loss_tensor(),
                            model.pred_tensor(),
                            model.error_tensor(),
                            model.summary_tensor()], feed_dict=feed_dict)

                    summary_writer.add_summary(summary, batch_count)

                    err = np.sum(errors) / errors.size
                    acc_train = 1 - err
                    loss = loss / conf.batch_size
                    loss_train = loss 
                    if abs(acc_train) > 2:
                        acc_train = None
                    if abs(loss_train) > 2:
                        loss_train = None

                    plotter.add_values(batch_count, loss_train=loss_train, acc_train=acc_train, redraw=False)

                    # Periodically save gifs of episodes, model parameters, and summary statistics.
                    if batch_count != 0:
                        if batch_count % conf.flush_plot_frequency == 0:
                            print("{}: Loss={} err={}".format(episode_count, loss, err))
                            summary_writer.flush()
                            plotter.redraw()

                        if batch_count % conf.model_save_frequency == 0:
                            saver.save(sess,conf.model_path+'/model-'+str(episode_count)+'.cptk')
                            print("Saved Model")


                        if conf.verify_dataset and batch_count % conf.verify_frequencey == 0:
                            err = verify(conf, sess, model, cls)
                            print("error on verification set: {}".format(err))

                if env != None and epoch != epochs - 1:
                    env.close()
                    env = None

            if epochs != 0:
                plotter.redraw()

            test(conf, sess, model, cls)

        except KeyboardInterrupt:
            print("W: interrupt received, stopping…")
        except EOFError:
            print("end of pickled file reached")
        except Exception as e:
            print('Exception {}'.format(e))
            traceback.print_exc()
        finally:
            if env != None:
                env.close()

def predict(sess, t, s, model, cls):
    t_input = np.expand_dims(process_frame(t, cls), axis=0)
    s_input = np.expand_dims(process_frame(s, cls), axis=0)

    feed_dict = {
                model.phase_tensor(): 1,
                model.network.s_input:s_input,
                model.network.t_input:t_input }

    pred_value = sess.run(model.pred_tensor(), feed_dict=feed_dict)
    return pred_value

def verify_err(sess, t, s, model, cls):
    t_input = np.expand_dims(process_frame(t, cls), axis=0)
    s_input = np.expand_dims(process_frame(s, cls), axis=0)

    feed_dict = {
                model.phase_tensor(): 1,
                model.network.s_input:s_input,
                model.network.t_input:t_input }

    errors = sess.run(model.error_tensor(), feed_dict=feed_dict)
    return np.sum(errors) / errors.size

def test(conf, sess, model, cls):
    test_iter = conf.test_iter

    print("testing... {} iterations".format(test_iter))
    env = UnityGame(conf, num_iter=test_iter, randomize=False)

    for episode_count in range(0, test_iter):
        env.new_episode()
        t = env.get_state().target_buffer()
        s = env.get_state().source_buffer()
        pred_value = predict(sess, t, s,  model, cls)
        true_value = np.expand_dims(model.true_value(env), axis=0)

        images = [t, s]
        err_str = model.error_str(true_value, pred_value)
        cap_texts = ["target:" + env.target_str(), "source:" + env.source_str()]
        cap_texts2 = [ err_str, "" ]

        if conf.test_steps == 0:
            make_jpg(conf, "test_set_", images, cap_texts, cap_texts2,  episode_count)
        else:
            for step in range(conf.test_steps):
                pred_value = pred_value[0]
                x = float(pred_value[0])
                y = float(pred_value[1])
                z = float(pred_value[2])
                env.take_prediction_step(x, y,z)
                image = env.get_state().source_buffer()
                images.append(image)
                pred_value = predict(sess, t, image, model, cls)
                true_value = np.expand_dims(model.true_value(env), axis=0)
                err_str = model.error_str(true_value, pred_value)
                cap_texts.append("step {}:{}".format(step+1, env.source_str()))
                cap_texts2.append(err_str)

            make_jpg(conf, "test_set_steps_", images, cap_texts, cap_texts2, episode_count)

    env.close()

def verify(conf, sess, model, cls):
    env = UnityGame(args, verify_iter, dataset = conf.verify_dataset, randomize=False)
    total_error = 0

    for episode_count in range(0, conf.verify_iter):
        env.new_episode()
        t = env.get_state().target_buffer()
        s = env.get_state().source_buffer()
        total_error += verify_err(sess, t, s,  model, cls)

    env.close()
    return total_error / conf.verify_iter

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
