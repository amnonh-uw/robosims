from robosims.unity import UnityGame
from util.util import *
from util.config import *
import pickle
import sys
import gzip

def gen_dataset(argv):
    args = parse_args(argv)
    args.gen_dataset = True
    conf = args.conf

    env = UnityGame(args)
    episode_count = 0
    
    if args.iter == 0:
        train_iter = 40000 + 1000
    else:
        train_iter = args.iter

    if args.dataset == None:
        print("--dataset required")
        
    print(conf)
    print("generating {} samples to {}".format(train_iter, args.dataset))
    with gzip.open(args.dataset, 'wb') as dataset:
        for i in range(0, train_iter):
            env.new_episode()
            pickle.dump(env, dataset, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    gen_dataset(sys.argv[1:])
