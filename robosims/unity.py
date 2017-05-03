import math
import random
import numpy as np
import robosims.server
from robosims.actions import *
import pickle
import gzip

class UnityGame:
    def __init__(self, args, port=0, start_unity = True):
        self.conf = args.conf
        if args.dataset == None or args.gen_dataset:
            self.controller = robosims.server.Controller(args.server_config)
            self.controller.start(port, start_unity)
            self.get_structure_info()
        else:
            self.controller = None
            self.dataset = gzip.open(args.dataset, 'rb')

        random.seed()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['controller']
        return state

    def close (self):
        if self.controller != None:
            self.controller.stop()
        else:
            self.dataset.close()

    def reset(self):
        self.controller.reset()

    def stop(self):
        if self.controller != None:
            self.controller.stop()

    def distance(self):
        delta = np.asarray(self.t_pos) - np.asarray(self.s_pos)
        dist = math.sqrt(np.sum(np.square(delta)))
        return dist

    def translation(self):
        delta = np.asarray(self.t_pos) - np.asarray(self.s_pos)
        return delta
        
    def one_close_enough(self, a, b, dist):
        return abs(a-b) < dist

    def close_enough(self):
        b = True

        b = b and self.one_close_enough(self.t_position['x'], self.s_position['x'], self.conf.close_enough_distance)
        b = b and self.one_close_enough(self.t_position['y'], self.s_position['y'], self.conf.close_enough_distance)
        b = b and self.one_close_enough(self.t_position['z'], self.s_position['z'], self.conf.close_enough_distance)
        b = b and self.one_close_enough(self.t_rotation['x'], self.s_rotation['x'], self.conf.close_enough_rotation)
        b = b and self.one_close_enough(self.t_rotation['y'], self.s_rotation['y'], self.conf.close_enough_rotation)
        b = b and self.one_close_enough(self.t_rotation['z'], self.s_rotation['z'], self.conf.close_enough_rotation)

        return b

    def new_episode(self):
        if self.controller == None:
            tmp_dict = pickle.load(self.dataset).__dict__

            self.__dict__.update(tmp_dict) 
        else:
            self.episode_finished = False
            self.gen_new_episode()
            self.collision = False
            self.action_counter = 0

    def get_state(self):
        return UnityState(self.s_frame, self.t_frame, self.collision)

    def is_episode_finished(self):
        return self.episode_finished

    def step(self, action):
        event = self.controller.step(action)
        controlCommand = event.metadata['controlCommand']
        self.collision = event.metadata['collided']
        if self.collision:
            self.collidedObjects = event.metadata['collidedObjects']
        agent = event.metadata['agent']
        self.s_position = agent['position']
        self.s_rotation = agent['rotation']
        # print("position {}".format(self.s_position))
        # print("rotation {}".format(self.s_rotation))
        self.extract_source_pose()
        return event

    def take_prediction_step(self, x, y, z):
        event = self.take_add_position_action(x, y, z)
        self.s_frame = event.frame

    def take_add_position_action(self, x, y, z):
        action = ActionBuilder.addPosition(x, y, z)
        return self.step(action)

    def take_add_rotation_action(self, rx, ry, rz):
        action = ActionBuilder.addRotation(rx, ry, rz)
        return self.step(action)

    def take_set_position_action(self, x, y, z):
        action = ActionBuilder.setPosition(x, y, z)
        return self.step(action)

    def take_set_rotation_action(self, rx, ry, rz):
        action = ActionBuilder.setRotation(rx, ry, rz)
        return self.step(action)

    def take_probe_action(self, x, y, z, d):
        action = ActionBuilder.probe(x, y, z, d)
        return self.step(action)

    def discrete_action_heuristic(self):
        pass

    def take_discrete_action(self, a):
        self.action_counter += 1

        if a == DiscreteAction.Nothing:
            event = self.take_add_position_action(0, 0, 0)
        elif a == DiscreteAction.Right:
            event = self.take_add_position_action(self.conf.discrete_action_distance, 0, 0)
        elif a == DiscreteAction.Left:
            event = self.take_add_position_action(-self.conf.discrete_action_distance, 0, 0)
        elif a == DiscreteAction.Up:
            event = self.take_add_position_action(0, self.conf.discrete_action_distance, 0)
        elif a == DiscreteAction.Down:
            event = self.take_add_position_action(0, -self.conf.discrete_action_distance, 0)
        elif a == DiscreteAction.Forward:
            event = self.take_add_position_action(0, 0, self.conf.discrete_action_distance)
        elif a == DiscreteAction.Backward:
            event = self.take_add_position_action(0, 0, -self.conf.discrete_action_distance)
        elif a == DiscreteAction.Clockwise:
            event = self.take_add_rotation_action(0, self.conf.discrete_action_rotation, 0)
        elif a == DiscreteAction.AntiClockwise:
            event = self.take_add_rotation_action(0, -self.conf.discrete_action_rotation, 0)

        self.s_frame = event.frame
        if self.action_counter == self.conf.max_episode_length:
            print("episode finished because of max length")
            self.episode_finished = True
        r = self.reward()
        print("{0}:action {1} reward {2}".format(self.action_counter, a, r))
        return r

    def take_continous_action(self, dx, dy, dz, dr):
        self.action_counter += 1
        print("{0}: action {1},{2},{3},{4}".format(self.action_counter, dx, dy, dz, dr))

        action = ActionBuilder.addPositionRotation(dx, dy, dz, 0, dr, 0)
        event = self.step(action)
        self.s_frame = event.frame
        if self.action_counter == self.conf.max_episode_length:
            print("episode finished because of max length")
            self.episode_finished = True
        r = self.reward()
        print("{0}: action {1},{2},{3},{4} reward {5}".format(self.action_counter, dx, dy, dz, dr, r))
        return r

    def reward(self):
        if self.collision:
            print("episode ended because of collision with {}".format(self.collidedObjects))
            self.episode_finished = True
            return self.conf.collision_reward - self.action_counter * self.conf.step_reward

        if self.close_enough():
            print("episode ended because close enough")
            self.episode_finished = True
            return self.conf.close_enough_reward

        # return self.conf.step_reward
        return -self.steps_left_heuristic()

    def steps_left_heuristic(self):
        steps = 0
        if self.conf.discrete_actions:
            delta = np.asarray(self.t_pos) - np.asarray(self.s_pos)
            for d in delta:
                steps += abs(int(d / self.conf.discrete_action_distance))

            delta = np.asarray(self.t_rot) - np.asarray(self.s_rot)
            for d in delta:
                steps += abs(int(d / self.conf.discrete_action_rotation))
            return steps
        else:
            return 4

    def next_step_heuristic(self):
        if self.conf.discrete_actions:
            h = self.conf.close_enough_reward + self.steps_left_heuristic() * self.conf.step_reward
            print("next_step_heuristic {}".format(h))
            return h
        else:
            return self.conf.close_enough_reward - self.stpes.left_heurisitc()

    def get_structure_info(self):
        # first figure out where the structure is and what its size is
        event = self.take_add_position_action(0, 0, 0)
        structure = event.metadata['structure']
        # print(structure)
        # position = structure['position']
        # extents = structure['extents']
        # self.structure_position = self.extract_position(position)
        # self.structure_extents = self.extract_position(extents)

        # print(self.structure_position)
        # print(self.structure_extents)

        # self.min_x = position['x'] - extents['x']
        # self.min_y = position['y'] - extents['y']
        # self.min_z = position['z'] - extents['z']


        # self.max_x = self.min_x + 2 * extents['x']
        # self.max_y = self.min_y + 2 * extents['y']
        # self.max_z = self.min_z + 2 * extents['z']

        print("overriding strucuture info")
        # min 8.0,0.8,10.6
        self.min_x = 8.0
        self.min_y = 0.8
        self.min_z = 10.6
        # max 14.0,3.9,13.4
        self.max_x = 14
        self.max_y = 3.9
        self.max_z = 13.4

    def valid_pose(self, x, y, z, r):
        #move to target
        event = self.take_set_position_action(x, y,z)
        if self.collision:
            return None


        # rotate to target
        event = self.take_set_rotation_action(0, r, 0)
        if self.collision:
            return None

        if self.conf.probe_distance != 0:
            event = self.take_probe_action(0, 0, 1, self.conf.probe_distance)
            if self.collision:
                return None

        return event

    def gen_new_episode(self):
        self.min_r = 0
        self.max_r = 360

        while True:
            self.s_x, self.s_y, self.s_z, self.s_r = self.random_source_pose()
            self.t_x, self.t_y, self.t_z, self.t_r = self.random_target_pose()
            self.t = self.valid_pose(self.t_x, self.t_y, self.t_z, self.t_r)
            if self.t is None:
                continue

            agent = self.t.metadata['agent']
            self.t_position = agent['position']
            self.t_rotation = agent['rotation']
            self.extract_target_pose()
            self.t_frame = self.t.frame

            self.s = self.valid_pose(self.s_x, self.s_y, self.s_z, self.s_r)
            if self.s is None:
                continue

            self.s_frame = self.s.frame
            break

        print("new episode {}{}-{}{}".format(self.s_pos, self.s_rot, self.t_pos, self.t_rot))

    def extract_position(self, p):
        return (self.grid_round(p['x'], self.conf.grid_distance),
                self.grid_round(p['y'], self.conf.grid_distance),
                self.grid_round(p['z'], self.conf.grid_distance))

    def extract_rotation(self, r):
        return (self.grid_round(r['x'], self.conf.grid_rotation),
                self.grid_round(r['y'], self.conf.grid_rotation),
                self.grid_round(r['z'], self.conf.grid_rotation))

    def extract_target_pose(self):
        self.t_pos = self.extract_position(self.t_position)
        self.t_rot = self.extract_rotation(self.t_rotation)

    def extract_source_pose(self):
        self.s_pos = self.extract_position(self.s_position)
        self.s_rot = self.extract_rotation(self.s_rotation)

    def random_source_pose(self):
        x = self.uniform_coord(self.min_x, self.max_x, self.conf.grid_distance)
        y = self.uniform_coord(self.min_y, self.max_y, self.conf.grid_distance)
        z = self.uniform_coord(self.min_z, self.max_z, self.conf.grid_distance)
        r = self.uniform_coord(self.min_r, self.max_r, self.conf.grid_rotation)

        return (x, y, z, r)

    def random_target_pose(self):
        x = self.uniform_delta(self.conf.max_distance_delta, self.s_x, self.min_x, self.max_x, self.conf.grid_distance)
        y = self.uniform_delta(self.conf.max_distance_delta, self.s_y, self.min_y, self.max_y, self.conf.grid_distance)
        z = self.uniform_delta(self.conf.max_distance_delta, self.s_z, self.min_z, self.max_z, self.conf.grid_distance)
        r = self.uniform_delta(self.conf.max_rotation_delta, self.s_r, self.min_r, self.max_r, self.conf.grid_rotation)

        return (x, y, z, r)

    def grid_round(self, g, grid):
        if grid is None:
            return g

        l = -math.log10(grid)
        li = int(l)
        if li == l:
            return round(g, li)
            
        r = g % grid
        g -= r
        if r >= grid/2:
             g += grid

        return g

    def uniform_coord(self, min_g, max_g, grid):
        g = np.random.uniform(min_g, max_g)
        return self.grid_round(g, grid)

    def uniform_delta(self, max_delta, g, min_g, max_g, grid):
        if g + max_delta <= max_g:
            max_g = g + max_delta
        if g - max_delta >= min_g:
            min_g = g - max_delta
        g = np.random.uniform(min_g, max_g)

        return self.grid_round(g, grid)

    def source_str(self):
        return str(self.s_pos) + str(self.s_rot)

    def target_str(self):
        return str(self.t_pos) + str(self.t_rot)

class UnityState:
    def __init__(self, s_frame, t_frame, collision):
        self.s_frame = s_frame
        self.t_frame = t_frame
        self.collision = collision

    def target_buffer(self):
        return self.t_frame

    def source_buffer(self):
        return self.s_frame

    def sensor_input(self):
        return [self.collision]
