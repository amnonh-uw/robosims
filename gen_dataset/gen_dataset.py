from robosims.unity import UnityGame, DatasetInfo
from util.util import *
from util.config import *
import pickle
import sys
import numpy as np
import random

def gen_dataset(argv):
    args = parse_args(argv)
    conf = args.conf
    conf.gen_dataset = True

    env = UnityGame(conf)
    episode_count = 0
    
    if args.iter == 0:
        if args.test_only:
            train_iter = 1000
        else:
            train_iter = 40000 + 1000
    else:
        train_iter = args.iter

    if args.dataset == None:
        print("--dataset required")
        
    print(conf)
    print("generating {} samples to {}".format(train_iter, args.dataset))
    data_file, idx_file  = dataset_files(args.dataset)
    index = np.zeros([train_iter], dtype=int)

    with open(data_file, 'wb') as data:
        for i in range(0, train_iter):
            env.new_episode()
            index[i] = data.tell()
            pickle.dump(env, data, pickle.HIGHEST_PROTOCOL)

    dataset_info = DatasetInfo(conf)
    with open(idx_file, 'wb') as idx:
        pickle.dump(index, idx, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset_info, idx, pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":
    gen_dataset(sys.argv[1:])
