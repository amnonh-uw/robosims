import argparse
import os
import sys
import multiprocessing
import threading
import tensorflow as tf
from a3c.worker import Worker
from a3c.ac_network.ac_network import AC_Network
from robosims.unity import UnityGame

def train(argv):
    parser = argparse.ArgumentParser(description='a3c train')
    parser.add_argument('--num_workers', type=int, help='number of workers', default=multiprocessing.cpu_count())
    parser.add_argument('--max_episode_length', type=int, help='maximum episode length', default=30)
    parser.add_argument('--episode_buffer_size', type=int, help='episode buffer size', default=30)
    parser.add_argument('--h_size', type=int, help='horizontal image size', default=400)
    parser.add_argument('--v_size', type=int, help='vertical image size', default=300)
    parser.add_argument('--channels', type=int, help='image channels', default=3)
    parser.add_argument('--sensors', type=int, help='sensor channels', default=1)
    parser.add_argument('--gamma', type=float, help='discount rate', default=0.99)
    parser.add_argument('--config', type=str, help='config file', default="")
    parser.add_argument('--load-model', dest='load_model', action='store_true')
    parser.add_argument('--no-load_model', dest='load_model', action='store_false')
    parser.add_argument('--initialize-weights', type=str, help='initialize weights', default="")
    parser.add_argument('--discrete_action_distance', type=float, default=0.01)
    parser.add_argument('--discrete_action_rotation', type=float, default=1)
    parser.add_argument('--close-enough-distance', type=float, default = 0.01)
    parser.add_argument('--close-enough-rotation', type=float, default = 1)
    parser.add_argument('--max-distance-delta', type=float, default = 0.03)
    parser.add_argument('--max-rotation-delta', type=float, default = 3)
    parser.add_argument('--collision-reward', type=int, default = -1000)
    parser.add_argument('--step-reward', type=int, default = -1)
    parser.add_argument('--close-enough-reward', type=int, default = 1000)
    parser.add_argument('--discrete-actions', dest='discrete_actions', action='store_true')
    parser.add_argument('--continous_actions', dest='discrete_actions', action='store_false')
    parser.set_defaults(load_model=False)
    parser.set_defaults(discrete_actions=False)

    args = parser.parse_args(argv)
    if args.discrete_actions:
        args.a_size = 9
    else:
        args.a_size = 4
    print(args)

    model_path = './model'

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    #Create a directory to save episode playback gifs to
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    with tf.device("/cpu:0"): 
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        master_network = AC_Network(args, 'global',None) # Generate global network
        workers = []
        # Create worker classes
        for i in range(args.num_workers):
            workers.append(Worker(i,UnityGame, args, AC_Network, trainer,model_path,global_episodes))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if args.load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            if args.initialize_weights != "":
                with sess.as_default(), sess.graph.as_default():
                    print("loading master network weights")
                    master_network.load(args.initialize_weights, sess)
        
        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(args,sess,coord,saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            worker_threads.append(t)
        coord.join(worker_threads)

if __name__ == "__main__":
    train(sys.argv[1:])
