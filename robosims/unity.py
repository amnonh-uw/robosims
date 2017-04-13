import math
import random
import numpy as np
import robosims.server
from robosims.actions import ActionBuilder

class UnityGame:
    def __init__(self, args, port=0, start_unity = True):
        self.controller = robosims.server.Controller(args.server_config)
        self.controller.start(port, start_unity)
        self.conf = args.conf

        self.get_structure_info()
        random.seed()

    def close (self):
        self.controller.stop()

    def reset(self):
        self.controller.reset()

    def stop(self):
        self.controller.stop()

    def distance(self):
        delta = np.asarray(self.t_pos) - np.asarray(self.s_pos)
        dist = math.sqrt(np.sum(np.square(delta)))
        return dist

    def direction(self):
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
        self.episode_finished = False
        self.gen_new_episode()
        self.collision = False
        self.action_counter = 0
        # print("starting new episode")

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

    def take_add_position_action(self, x, y, z):
        action = ActionBuilder.addPosition(x, y, z)
        return self.step(action)

    def take_add_rotation_action(self, rx, ry, rz):
        action = ActionBuilder.addRotation(rx, ry, rz)
        return self.step(action)

    def take_set_position_action(self, x, y, z):
        action = ActionBuilder.setPosition(x, y, z)
        return self.step(action)

    def take_discrete_action(self, a):
        self.action_counter += 1

        if a == 0:
            event = self.take_add_position_action(0, 0, 0)
        elif a == 1:
            event = self.take_add_position_action(self.conf.discrete_action_distance, 0, 0)
        elif a == 2:
            event = self.take_add_position_action(-self.conf.discrete_action_distance, 0, 0)
        elif a == 3:
            event = self.take_add_position_action(0, self.conf.discrete_action_distance, 0)
        elif a == 4:
            event = self.take_add_position_action(0, -self.conf.discrete_action_distance, 0)
        elif a == 5:
            event = self.take_add_position_action(0, 0, self.conf.discrete_action_distance)
        elif a == 6:
            event = self.take_add_position_action(0, 0, -self.conf.discrete_action_distance)
        elif a == 7:
            event = self.take_add_rotation_action(0, self.conf.discrete_action_rotation, 0)
        elif a == 8:
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

        return self.conf.step_reward

    def reward_heuristic(self):
        steps = 0
        if self.conf.discrete_actions:
            delta = np.asarray(self.t_pos) - np.asarray(self.s_pos)
            for d in delta:
                steps += abs(int(d / self.conf.discrete_action_distance))

            delta = np.asarray(self.t_rot) - np.asarray(self.s_rot)
            for d in delta:
                steps += abs(int(d / self.conf.discrete_action_rotation))

            print("heuristic steps {}".format(steps))
            return self.conf.close_enough_reward + steps * self.conf.step_reward
        else:
            return self.conf.close_enough_reward - 4

    def get_structure_info(self):
        # first figure out where the structure is and what its size is
        action = ActionBuilder.addPosition(0, 0, 0)
        event = self.step(action)
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


    def gen_new_episode(self):
        self.min_r = 0
        self.max_r = 360

        while True:
            self.s_x, self.s_y, self.s_z, self.s_r = self.random_source_pose()
            self.t_x, self.t_y, self.t_z, self.t_r = self.random_target_pose()

            # print("x: {} vs {}".format(self.s_x, self.t_x))
            # print("y: {} vs {}".format(self.s_y, self.t_y))
            # print("z: {} vs {}".format(self.s_z, self.t_z))
            # print("r: {} vs {}".format(self.s_r, self.t_r))

            # move to target
            action = ActionBuilder.setPosition(self.t_x, self.t_y, self.t_z)
            self.step(action)
            if self.collision:
                print('collision on target, retrying')
                continue

            # rotate to target
            action = ActionBuilder.setRotation(0, self.t_r, 0)
            self.t = self.step(action)
            if self.collision:
                print('collision on target, retrying')
                continue

            agent = self.t.metadata['agent']
            self.t_position = agent['position']
            self.t_rotation = agent['rotation']
            self.extract_target_pose()
            self.t_frame = self.t.frame

            # input("Hit enter to accept target")
            break

        while True:
            # move to source
            action = ActionBuilder.setPosition(self.s_x, self.s_y, self.s_z)
            self.step(action)
            if self.collision:
                print('collision on source, retrying')
                continue

            # rotate to source
            action = ActionBuilder.setRotation(0, self.s_r, 0)
            self.s = self.step(action)
            if self.collision:
                print('collision on source, retrying')
                continue

            self.s_frame = self.s.frame
            # input("Hit enter to accept source")
            break

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
