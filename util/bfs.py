from util.config import *
from robosims.unity import *
import random
import sys

class BFS:
    def __init__(self, conf, env):
        self.visited = set()   
        self.collided_objects = set()
        self.min_x = 1000
        self.min_y = 1000
        self.min_z = 1000
        self.max_x = -1000
        self.max_y = -1000
        self.max_z = -1000

        self.env = env
        self.conf = conf
    
    def search(self, v, collided = False):
        if v in self.visited:
            return

        if collided:
            for o in self.env.collidedObjects:
                self.collided_objects.add(o)
            return

        self.visited.add(v)
        x = v[0]
        y = v[1]
        z = v[2]

        if self.min_x > x:
            self.min_x = x
        if self.min_y > y:
            self.min_y = y
        if self.min_z > z:
            self.min_z = z

        if self.max_x < x:
            self.max_x = x
        if self.max_y < y:
            self.max_y = y
        if self.max_z < z:
            self.max_z = z

        self.env.take_add_position_action(self.conf.bfs_grid_x, 0, 0)
        self.search(self.env.s_pos, self.env.collision)
        self.env.take_set_position_action(x, y, z)

        self.env.take_add_position_action(-self.conf.bfs_grid_x, 0, 0)
        self.search(self.env.s_pos, self.env.collision)
        self.env.take_set_position_action(x, y, z)

        self.env.take_add_position_action(0, self.conf.bfs_grid_y, 0)
        self.search(self.env.s_pos, self.env.collision)
        self.env.take_set_position_action(x, y, z)

        self.env.take_add_position_action(0, -self.conf.bfs_grid_y, 0)
        self.search(self.env.s_pos, self.env.collision)
        self.env.take_set_position_action(x, y, z)

        self.env.take_add_position_action(0, 0, self.conf.bfs_grid_z)
        self.search(self.env.s_pos, self.env.collision)
        self.env.take_set_position_action(x, y, z)

        self.env.take_add_position_action(0, 0, -self.conf.bfs_grid_z)
        self.search(self.env.s_pos, self.env.collision)
        self.env.take_set_position_action(x, y, z)


    def sample(self):
        return random.sample(self.visited, 1)

    def sample2(self):
        return random.sample(self.visited, 2)


def run_bfs(argv):
    args = parse_args(argv)
    env = UnityGame(args)
    bfs = BFS(args.conf, env)
    lim = sys.getrecursionlimit();
    sys.setrecursionlimit(1000 * lim);
    bfs.search(env.s_pos)
    print("{} items in the set".format(len(bfs.visited)))
    print("collided objects: {}".format(bfs.collided_objects))
    print("min {},{},{}".format(bfs.min_x, bfs.min_y, bfs.min_z))
    print("max {},{},{}".format(bfs.max_x, bfs.max_y, bfs.max_z))
    env.close()

if __name__ == "__main__":
    run_bfs(sys.argv[1:])
