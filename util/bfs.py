from util.config import *
from robosims.unity import *
import random
import sys

class BFS:
    def __init__(self, conf, env):
        self.visited = set()   
        self.min_x = 1000
        self.min_y = 1000
        self.min_z = 1000
        self.max_x = -1000
        self.max_y = -1000
        self.max_z = -1000

        self.env = env
        self.conf = conf
    
    def search(self, v, collided = False):
        if v not in self.visited and not collided:
            # print("bfs adding {}".format(v))
            self.visited.add(v)
        else:
            # print("bfs not adding {} collision {}".format(v, collided))
            return

        if self.min_x > v[0]:
            self.min_x = v[0]
        if self.min_y > v[1]:
            self.min_y = v[1]
        if self.min_z > v[2]:
            self.min_z = v[2]

        if self.max_x < v[0]:
            self.max_x = v[0]
        if self.max_y < v[1]:
            self.max_y = v[1]
        if self.max_z < v[2]:
            self.max_z = v[2]

        self.env.take_add_position_action(self.conf.bfs_grid_x, 0, 0)
        self.search(self.env.s_pos, self.env.collision)
        self.env.take_add_position_action(-self.conf.bfs_grid_x, 0, 0)
        self.search(self.env.s_pos, self.env.collision)
        self.env.take_add_position_action(0, self.conf.bfs_grid_y, 0)
        self.search(self.env.s_pos, self.env.collision)
        self.env.take_add_position_action(0, -self.conf.bfs_grid_y, 0)
        self.search(self.env.s_pos, self.env.collision)
        self.env.take_add_position_action(0, 0, self.conf.bfs_grid_z)
        self.search(self.env.s_pos, self.env.collision)
        self.env.take_add_position_action(0, 0, -self.conf.bfs_grid_z)
        self.search(self.env.s_pos, self.env.collision)

    def sample(self):
        return random.sample(self.visited, 1)

    def sample2(self):
        return random.sample(self.visited, 2)


def run_bfs(argv):
    args = parse_args(argv)
    env = UnityGame(args)
    bfs = BFS(args.conf, env)
    lim = sys.getrecursionlimit();
    sys.setrecursionlimit(10 * lim);
    bfs.search(env.s_pos)
    print("{} items in the set".format(len(bfs.visited)))
    print("min {},{},{}".format(bfs.min_x, bfs.min_y, bfs.min_z))
    print("max {},{},{}".format(bfs.max_x, bfs.max_y, bfs.max_z))
    env.close()

if __name__ == "__main__":
    run_bfs(sys.argv[1:])
