import math
import random
import numpy as np
import robosims.server
from robosims.actions import *
import pickle
from threading import Lock
from PIL import Image

version = 2

class DatasetInfo:
    def __init__(self, conf):
        self.version = version
        self.max_distance_delta = conf.max_distance_delta
        self.max_rotation_delta = conf.max_rotation_delta
        self.too_far_prob = conf.too_far_prob
        self.close_enough_prob = conf.close_enough_prob
    
    def check(self, conf):
        if self.version != version:
            raise ValueError("version {} inconsistent with index {}".
                            format(version, self.version))

        if self.too_far_prob != conf.too_far_prob:
            raise ValueError("too_far_prob {} inconsistent with index {}".
                            format(conf.too_far_prob, self.too_far_prob))

        if self.close_enough_prob != conf.close_enough_prob:
            raise ValueError("close_enough_prob {} inconsistent with index {}".
                            format(conf.close_enough_prob, self.close_enough_prob))

        if self.max_distance_delta != conf.max_distance_delta:
            raise ValueError("max_distance_delta {} inconsistent with index {}".
                            format(conf.max_distance_delta, self.max_distance_delta))
        
        if self.max_rotation_delta != conf.max_rotation_delta:
            raise ValueError("max_rotation_delta {} inconsistent with index {}".
                            format(conf.max_rotation_delta, self.max_rotation_delta))

class UnityGame:
    def __init__(self, conf, port=0, start_unity = True, dataset=None, num_iter=0, randomize=True):
        self.conf = conf
        random.seed()

        if dataset == None or conf.gen_dataset:
            self.dataset = None
            self.controller = robosims.server.Controller(conf.server_config)
            self.controller.start(port, start_unity)
            self.get_structure_info()
        else:
            self.controller = None
            data_file, idx_file  = self.dataset_files(dataset)

            with open(idx_file, "rb") as idx:
                self.index = pickle.load(idx)
                if num_iter != 0:
                    if num_iter > self.index.size:
                            raise ValueError("num_iter {} inconsistent with index size {}".
                                format(num_iter, self.index.size))

                    self.index = self.index[:num_iter]

                    dataset_info = pickle.load(idx)
                    dataset_info.check(conf)

                if randomize:
                    np.random.shuffle(self.index)
                self.episode_counter = 0

            self.dataset = open(data_file, "rb")
            if self.dataset == None:
                print("can't open {}".format(data_file))
                raise ValueError("can't open " + data_file)

            self.dataset_lock = Lock()

    @staticmethod
    def remove_private_members(tmp_dict):
        tmp_dict.pop('conf', 0)
        tmp_dict.pop('dataset', 0)
        tmp_dict.pop('index', 0)
        tmp_dict.pop('episode_counter', 0)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['controller']
        return state

    def dataset_files(self, dataset):
        idx_file = dataset + ".idx"
        data_file = dataset + ".data"

        return data_file, idx_file

    def close (self):
        if self.dataset != None:
            self.dataset.close()

        if self.controller != None:
            self.controller.stop()

    def reset(self):
        self.controller.reset()

    def stop(self):
        if self.controller != None:
            self.controller.stop()

    def distance(self):
        delta = np.asarray(self.t_pos) - np.asarray(self.s_pos)
        dist = math.sqrt(np.sum(np.square(delta)))
        return dist

    def translation(self, dims=3):
        delta_xyz = np.asarray(self.t_pos) - np.asarray(self.s_pos)
        delta_pqr = np.asarray(self.t_rot) - np.asarray(self.s_rot)

        if dims == 1:
            return np.reshape(np.asarray(np.linalg.norm(delta_xyz)), [1])
        if dims == 3:
            return(delta_xyz)
        if dims == 4:
            r = delta_pqr[1]
            return np.append(delta_xyz, r)

        raise ValueError("dim must be 1, 3 or 4")

    def get_class(self, dims=3):
        if dims == 0:
            if self.all_close_enough():
                cls = 1
            elif self.too_far_episode:
                cls = 2
            else:
                cls = 0

        else:
            delta = self.translation(dims=dims)
            cls = np.argmax(delta)
            if delta[cls] < 0:
                cls += dims

            if self.all_close_enough():
                cls = 2 * dims

        return int(cls)
        
    def one_close_enough(self, a, b, dist):
        return abs(a-b) < dist

    def all_close_enough(self):
        b = True

        b = b and self.one_close_enough(self.t_position['x'], self.s_position['x'], self.conf.close_enough_distance)
        b = b and self.one_close_enough(self.t_position['y'], self.s_position['y'], self.conf.close_enough_distance)
        b = b and self.one_close_enough(self.t_position['z'], self.s_position['z'], self.conf.close_enough_distance)
        b = b and self.one_close_enough(self.t_rotation['x'], self.s_rotation['x'], self.conf.close_enough_rotation)
        b = b and self.one_close_enough(self.t_rotation['y'], self.s_rotation['y'], self.conf.close_enough_rotation)
        b = b and self.one_close_enough(self.t_rotation['z'], self.s_rotation['z'], self.conf.close_enough_rotation)

        return b

    def new_episode(self, episode_num=None):
        if self.controller == None:
            if self.episode_counter == self.index.size:
                raise ValueError("number of episodes in data set exceeded")

            if episode_num == None:
                episode_num = self.episode_counter
                self.episode_counter += 1

            with self.dataset_lock:
                self.dataset.seek(self.index[episode_num])
                tmp_dict = pickle.load(self.dataset).__dict__

            self.remove_private_members(tmp_dict)
            self.__dict__.update(tmp_dict) 
        else:
            close_enough = False
            too_far = False

            r = random.random()
            if r < self.conf.close_enough_prob:
                close_enough = True
            elif r < (self.conf.too_far_prob + self.conf.close_enough_prob):
                too_far = True

            self.gen_new_episode(close_enough = close_enough, too_far = too_far)
            self.close_enough_episode = close_enough
            self.too_far_episode = too_far

        self.episode_finished = False
        self.collision = False
        self.action_counter = 0

    def get_state(self):
        return UnityState(self.s_frame, self.t_frame, self.s_frame_depth, self.t_frame_depth, self.t_to_s_frame_flow, self.collision)

    def is_episode_finished(self):
        return self.episode_finished

    def step(self, action):
        event = self.controller.step(action)
        self.collision = event.metadata['collided']
        if self.collision:
            self.collidedObjects = event.metadata['collidedObjects']
        agent = event.metadata['agent']
        self.s_position = agent['position']
        self.s_rotation = agent['rotation']
        self.extract_source_pose()
        return event

    def take_prediction_step(self, step):
        x = y = z = r = 0
        dims = step.size
        x = float(step[0])
        if dims > 1:
            y = float(step[1])
        if dims > 2:
            z = float(step[2])
        if dims > 3:
            r = float(step[3])
        event = self.take_add_position_action(x, y, z)
        if r != 0:
            event = self.take_add_rotation_action(0, r, 0)
        
        self.s_frame = event.frame

    def take_add_position_action(self, x, y, z):
        action = ActionBuilder.addPosition(x, y, z)
        return self.step(action)

    def take_add_rotation_action(self, rx, ry, rz):
        action = ActionBuilder.addRotation(rx, ry, rz)
        return self.step(action)

    def take_add_action(self, x, y, z, rx, ry, rz):
        action = ActionBuilder.Add(x, y, z, rx, ry, rz)
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

    def take_set_action(self, x, y, z, rx, ry, rz):
        action = ActionBuilder.Set(x, y, z, rx, ry, rz)
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

        if self.all_close_enough():
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
        #move and rotate to target
        move_event = self.take_set_action(x,y,z, 0, r, 0)
        if self.collision:
            return None

        if self.conf.probe_distance != 0:
            event = self.take_probe_action(0, 0, 1, self.conf.probe_distance)
            if self.collision:
                return None

        agent = move_event.metadata['agent']
        pos = agent['position']
        rot = agent['rotation']

        if r >= 360:
            r -= 360
        if r < 0:
            r += 360

        assert abs(pos['x'] - x) < 0.01, "x %f, pos[x] %f" %  (x, pos['x'])
        assert abs(pos['y'] - y) < 0.01, "y %f, pos[y] %f" %  (x, pos['y'])
        assert abs(pos['z'] - z) < 0.01, "z %f, pos[z] %f" %  (x, pos['z'])
        assert abs(rot['y'] - r) < 0.01, "r %f, pos[r] %f" %  (x, pos['r'])

        return move_event

    def gen_new_episode(self, close_enough = False, too_far = False):
        self.min_r = 0
        self.max_r = 360

        while True:
            self.s_x, self.s_y, self.s_z, self.s_r = self.random_source_pose()
            self.t_x, self.t_y, self.t_z, self.t_r = self.random_target_pose(close_enough=close_enough, too_far=too_far)
            self.t = self.valid_pose(self.t_x, self.t_y, self.t_z, self.t_r)
            if self.t is None:
                continue

            agent = self.t.metadata['agent']
            self.t_position = agent['position']
            self.t_rotation = agent['rotation']
            self.extract_target_pose()
            self.t_frame = self.t.frame
            self.t_frame_depth = self.t.frame_depth

            self.s = self.valid_pose(self.s_x, self.s_y, self.s_z, self.s_r)
            if self.s is None:
                continue

            self.s_frame = self.s.frame
            self.s_frame_depth = self.s.frame_depth
            self.t_to_s_frame_flow = self.s.frame_flow
            break

        if not too_far:
            trans = self.translation(dims=4)

            assert abs(trans[0]) - self.conf.max_distance_delta < 0.01, "x %f > max %f" % (trans[0], self.conf.max_distance_delta)
            assert abs(trans[1]) - self.conf.max_distance_delta < 0.01, "y %f > max %f" % (trans[1], self.conf.max_distance_delta)
            assert abs(trans[2]) - self.conf.max_distance_delta < 0.01, "z %f > max %f" % (trans[2], self.conf.max_distance_delta)
#            assert abs(trans[3]) -  self.conf.max_rotation_delta < 0.01, "r %f > max %f" % (trans[3], self.conf.max_rotation_delta)

        # print("new episode {}{}-{}{}".format(self.s_pos, self.s_rot, self.t_pos, self.t_rot))

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

    def random_target_pose(self, close_enough = False, too_far = False):
        if too_far and close_enough:
            raise ValueError("too_far and close_enough cannot both be True")

        if close_enough:
            distance_range = self.conf.close_enough_distance
            rotation_range = self.conf.close_enough_rotation
        else:
            distance_range = self.conf.max_distance_delta
            rotation_range = self.conf.max_rotation_delta
            
        x = self.uniform_delta(distance_range, self.s_x, self.min_x, self.max_x, self.conf.grid_distance)
        y = self.uniform_delta(distance_range, self.s_y, self.min_y, self.max_y, self.conf.grid_distance)
        z = self.uniform_delta(distance_range, self.s_z, self.min_z, self.max_z, self.conf.grid_distance)
        r = self.uniform_delta(rotation_range, self.s_r, self.min_r, self.max_r, self.conf.grid_rotation)

        if too_far:
            r = 180

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
        s = self.s_pos
        s = s + (self.s_rot[1],)
        return str(s)

    def target_str(self):
        t = self.t_pos
        t = t + (self.t_rot[1],)
        return str(t)

class UnityState:
    def __init__(self, s_frame, t_frame, s_frame_depth, t_frame_depth, t_to_s_frame_flow, collision):
        self.s_frame = s_frame
        self.t_frame = t_frame
        self.s_frame_depth = s_frame_depth
        self.t_frame_depth = t_frame_depth
        self.t_to_s_frame_flow = t_to_s_frame_flow
        self.collision = collision

    def target_buffer(self):
        return self.t_frame

    def source_buffer(self):
        return self.s_frame

    def target_depth_buffer(self):
        return self.t_frame_depth

    def source_depth_buffer(self):
        return self.s_frame_depth

    def sensor_input(self):
        return [self.collision]
