import math
import random
import numpy as np
import robosims.server
from robosims.actions import ActionBuilder

class UnityGame:
    def __init__(self, args, port=0, start_unity = True):
        self.controller = robosims.server.Controller(args.config)
        self.controller.start(port, start_unity)
        self.args = args

        self.get_structure_info()
        random.seed()
    def close (self):
        print("stopping controller")
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

        b = b and self.one_close_enough(self.t_position['x'], self.s_position['x'], self.args.close_enough_distance)
        b = b and self.one_close_enough(self.t_position['y'], self.s_position['y'], self.args.close_enough_distance)
        b = b and self.one_close_enough(self.t_position['z'], self.s_position['z'], self.args.close_enough_distance)
        b = b and self.one_close_enough(self.t_rotation['x'], self.s_rotation['x'], self.args.close_enough_rotation)
        b = b and self.one_close_enough(self.t_rotation['y'], self.s_rotation['y'], self.args.close_enough_rotation)
        b = b and self.one_close_enough(self.t_rotation['z'], self.s_rotation['z'], self.args.close_enough_rotation)

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
        # print("Taking step {}".format(action));
        event = self.controller.step(action)
        controlCommand = event.metadata['controlCommand']
        # print("step control Command returned {}".format(controlCommand))
        self.collision = event.metadata['collided']
        if self.collision:
            self.collidedObjects = event.metadata['collidedObjects']
        agent = event.metadata['agent']
        self.s_position = agent['position']
        self.s_rotation = agent['rotation']
        # print("position {}".format(self.s_position))
        # print("rotation {}".format(self.s_rotation))
        self.extract_source_position()
        return event

    def take_discrete_action(self, a):
        self.action_counter += 1

        if a == 0:
            action = ActionBuilder.addPosition(0, 0, 0)
        elif a == 1:
            action = ActionBuilder.addPosition(self.args.discrete_action_distance, 0, 0)
        elif a == 2:
            action = ActionBuilder.addPosition(-self.args.discrete_action_distance, 0, 0)
        elif a == 3:
            action = ActionBuilder.addPosition(0, self.args.discrete_action_distance, 0)
        elif a == 4:
            action = ActionBuilder.addPosition(0, -self.args.discrete_action_distance, 0)
        elif a == 5:
            action = ActionBuilder.addPosition(0, 0, self.args.discrete_action_distance)
        elif a == 6:
            action = ActionBuilder.addPosition(0, 0, -self.args.discrete_action_distance)
        elif a == 7:
            action = ActionBuilder.addRotation(0, self.args.discrete_action_rotation, 0)
        elif a == 8:
            action = ActionBuilder.addRotation(0, -self.args.discrete_action_rotation, 0)

        event = self.step(action)
        self.s_frame = event.frame
        if self.action_counter == self.args.max_episode_length:
            print("episode finished because of max length")
            self.episode_finished = True
        r = self.reward();
        print("{0}:action {1} reward {2}\ns {3},{4}\nt {5},{6} ".format(self.action_counter, a, r, self.s_pos, self.s_rot, self.t_pos, self.t_rot))
        return r

    def take_continous_action(self, dx, dy, dz, dr):
        self.action_counter += 1
        print("{0}: action {1},{2},{3},{4}".format(self.action_counter, dx, dy, dz, dr))

        action = ActionBuilder.addPositionRotation(dx, dy, dz, 0, dr, 0)
        event = self.step(action)
        self.s_frame = event.frame
        if self.action_counter == self.args.max_episode_length:
            print("episode finished because of max length")
            self.episode_finished = True
        r = self.reward()
        print("{0}: action {1},{2},{3},{4} reward {5}".format(self.action_counter, dx, dy, dz, dr, r))
        return r

    def reward(self):
        if self.collision:
            print("episode ended because of collision with {}".format(self.collidedObjects))
            self.episode_finished = True
            return self.args.collision_reward - self.action_counter * self.args.step_reward

        if self.close_enough():
            print("episode ended because close enough")
            self.episode_finished = True
            return self.args.close_enough_reward

        return self.args.step_reward

    def get_structure_info(self):
        # first figure out where the structure is and what its size is
        action = ActionBuilder.addPosition(0, 0, 0)
        event = self.step(action)
        structure = event.metadata['structure']
        position = structure['position']
        size = structure['size']

        # print(position)
        # print(size)

        self.minx = position['x'] + 1
        self.miny = position['y'] + 1
        self.minz = position['z'] + 1

        # self.maxx = self.minx + size['x']
        # self.maxy = self.miny + size['y']
        # self.maxz = self.minz + size['z']

        print("overriding strucuture info")
        self.maxx = self.minx + 1
        self.maxy = self.miny + 0.01
        self.maxz = self.minz + 1

    # todo
    # need discrete version
    # need to figure out min and max for coordinates


    def gen_new_episode(self):
        self.minr = 0
        self.maxr = 360

        while True:
            self.s_x, self.s_y, self.s_z, self.s_r = self.random_pose()
            self.t_x, self.t_y, self.t_z, self.t_r = self.random_pose_delta()
            self.t_x += self.s_x
            self.t_y += self.s_y
            self.t_z += self.s_z
            self.t_r += self.s_r

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
            self.extract_target_position()
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

    def extract_target_position(self):
        self.t_pos = (round(self.t_position['x'], 2),
                      round(self.t_position['y'], 2),
                      round(self.t_position['z'], 2))
        self.t_rot = (round(self.t_rotation['x'], 2),
                      round(self.t_rotation['y'], 2),
                      round(self.t_rotation['z'], 2))

    def extract_source_position(self):
        self.s_pos = (round(self.s_position['x'], 2),
                      round(self.s_position['y'], 2),
                      round(self.s_position['z'], 2))
        self.s_rot = (round(self.s_rotation['x'], 2),
                      round(self.s_rotation['y'], 2),
                      round(self.s_rotation['z'], 2))

    def random_pose(self):
        x = round(random.uniform(self.minx, self.maxx), 2)
        y = round(random.uniform(self.miny, self.maxy), 2)
        z = round(random.uniform(self.minz, self.maxz), 2)
        r = round(random.uniform(self.minr, self.maxr))

        return (x, y, z, r)

    def random_pose_delta(self):
        x = round(random.uniform(-self.args.max_distance_delta, self.args.max_distance_delta), 2)
        y = round(random.uniform(-self.args.max_distance_delta, self.args.max_distance_delta), 2)
        z = round(random.uniform(-self.args.max_distance_delta, self.args.max_distance_delta), 2)
        r = round(random.uniform(-self.args.max_rotation_delta, self.args.max_rotation_delta), 2)

        return (x, y, z, r)

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
