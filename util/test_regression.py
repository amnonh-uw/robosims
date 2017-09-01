import math
import numpy as np
import os
from robosims.unity import UnityGame
from util.util import *
from util.laplotter import *

def test(conf, model, predict, steps = 0):
    test_iter = conf.test_iter

    if steps == 0:
        testset = conf.testset
        name = testset
    else:
        testset = None

    if name is None:
        name = "live game"

    print("testing... {} iterations from {}".format(test_iter, name))

    env = UnityGame(conf, dataset=testset, num_iter=test_iter, randomize=False)

    def empty_strings(n):
        l = []
        for k in range(n):
            l.append("")
        return l

    def empty_colors(n):
        l = []
        for k in range(n):
            l.append("white")
        return l

    for episode_count in range(0, test_iter):
        env.new_episode()
        t = env.get_state().target_buffer()
        s = env.get_state().source_buffer()
        pred_value = predict(t, s,  model)
        true_value = np.expand_dims(model.true_value(env), axis=0)

        images = [t, s]

        cap_texts = [("target:" + env.target_str())]
        cap_colors = ["white"]
        err_captions, err_colors, errs_absolute  = model.error_captions(true_value, pred_value)

        cap_texts.extend(err_captions)
        cap_colors.extend(err_colors)
        images_cap_texts = [cap_texts]
        images_cap_colors = [cap_colors]

        cap_texts = ["source:" + env.source_str()]
        cap_texts.extend(empty_strings(len(err_captions)))
        cap_colors = ["white"]
        cap_colors.extend(empty_colors(len(err_captions)))
        images_cap_texts.append(cap_texts)
        images_cap_colors.append(cap_colors)

        if steps == 0:
            print(" ".join(err_captions))
            if episode_count == 0:
                total_errs = errs_absolute
            else:
                total_errs = [x+y for x,y in zip(total_errs, errs_absolute)]

            make_jpg(conf, "test_set_", images, images_cap_texts,  images_cap_colors, episode_count)
        else:
            for step in range(steps):
                pred_value = np.squeeze(pred_value)
                model.take_prediction_step(env, pred_value)
                image = env.get_state().source_buffer()
                images.append(image)

                pred_value = predict(t, image, model)
                true_value = np.expand_dims(model.true_value(env), axis=0)
                err_captions, err_colors, _ = model.error_captions(true_value, pred_value)
                cap_texts = [("step {}:{}".format(step+1, env.source_str()))]
                cap_texts.extend(err_captions)
                cap_colors = ["white"]
                cap_colors.extend(err_colors)
                images_cap_texts.append(cap_texts)
                images_cap_colors.append(cap_colors)

            make_jpg(conf, "test_set_steps_", images, images_cap_texts, images_cap_colors, episode_count)

    if steps == 0:
        print("avg absolute errors: " + " ".join(["{0:.2f}".format(x / episode_count) for x in total_errs]))

    env.close()

def regression_predict(sess, cls, t, s, model):
    t_input = np.expand_dims(process_frame(t, cls), axis=0)
    s_input = np.expand_dims(process_frame(s, cls), axis=0)

    feed_dict = {
                model.phase_tensor(): 1,
                model.network.s_input:s_input,
                model.network.t_input:t_input }

    pred_value = sess.run(model.pred_tensor(), feed_dict=feed_dict)
    return pred_value
