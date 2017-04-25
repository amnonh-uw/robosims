import argparse
import os
import sys
import multiprocessing
import threading
import tensorflow as tf
import importlib
from a3c.worker import Worker
from a3c.ac_network import AC_Network
from robosims.unity import UnityGame
from util.config import *
from util.util import *

def train(argv):
    args = parse_args(argv)
    conf = args.conf

    if conf.discrete_actions:
        conf.a_size = 9
    else:
        conf.a_size = 4

    print(args)

    mod = importlib.import_module(conf.base_class)
    conf.cls = getattr(mod, conf.base_class)
    conf.cls_data = sys_path_find(conf.base_class + ".npy")
    if conf.cls_data == None:
        print("can't find data file for class {}".format(conf.base_class))
        exit()

    make_dirs('a3c', conf)

    tf.reset_default_graph()

    with tf.device("/cpu:0"): 
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)
        master_network = AC_Network(conf, 'main',None) # Generate global network
        workers = []
        # Create worker classes
        for i in range(conf.num_workers):
            workers.append(Worker(i,UnityGame, args, AC_Network, trainer,global_episodes))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if args.load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        elif args.partial_load_model != None:
            sess.run(tf.global_variables_initializer())
            optimistic_restore(sess, args.partial_load_model)
        else:
            sess.run(tf.global_variables_initializer())
            with sess.as_default(), sess.graph.as_default():
                print("loading master network weights")
                master_network.load(conf.cls_data, sess)
        
        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(conf,sess,coord,saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            worker_threads.append(t)
        coord.join(worker_threads)

if __name__ == "__main__":
    train(sys.argv[1:])
