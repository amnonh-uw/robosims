import random
import math
from robosims.actions import ActionBuilder



class Model(object):
    # Generate random point x_0
    # move to it
    # take a snapshop i_0
    # move to a random point at most max_distance away, x_1
    # take a snapshot i_1
    # caluclate a first step from x_0 to x_1, s
    # output i_0, i_1, s

    # todo
    # need discrete version
    # need to figure out min and max for coordinates
    # need to verify that second point is in the box

    def __init__(self):
        self.state = self.state0
        self.max_distance = 10.
        self.reset = True
        self.minx = self.miny = self.minz = 0
        self.maxx = self.maxx = self.maxz = 1
        self.minr = 0
        self.maxr = 360

    def next_action(self, event):
        return self.state(event)

    def reset(self):
        self.state = self.state0
        self.reset = True

    def state0(self, event):
        if self.reset:
            self.reset = False
        else:
            self.event1 = event
            print('saving sample information here')

        self.x0, self.y0, self. z0, self.r0 = self.random_pose()
        self.state = self.state0_rotate
        return ActionBuilder.setPosition(self.x0, self.y0, self.z0)

    def state0_rotate(self, event):
        self.state = self.state1
        return ActionBuilder.setRotation(self.r0, 0, 0)

    def state1(self, event):
        self.event0 = event
        self.x1, self.y1, self. z1, self.r1 = self.random_pose(self.max_distance)
        self.state = self.state1_rotate
        return ActionBuilder.setPosition(self.x1, self.y1, self.z1)

    def state1_rotate(self, event):
        self.state = self.state1
        return ActionBuilder.setRotation(self.r1, 0, 0)

    def random_pose(self, max_distance=0):
        if max_distance == 0:
            x = random.uniform(self.minx, self.maxx)
            y = random.uniform(self.miny, self.maxy)
            z = random.uniform(self.minz, self.maxz)
            r = random.uniform(self.minr, self.maxr)

            return (x, y, z, r)
        else:
            # generate random distance^2
            # generate three markers between 0 and distance^2
            # calculate dx_i^2
            # generate random signs and calculate dx_i

            distance = random.uniform(0, max_distance)
            dsq = distance * distance
            d = []
            for i in range(3):
                d.append(random.uniform(0, dsq))
            d.sort()
            c = []
            for i in range(3):
                x = dsq - d.pop()
                c.append(math.sqrt(x) * random.shuffle((-1, +1)))
                dsq -= x
            c.append(dsq * random.shuffle((-1, +1))))
            random.shuffle(c)

            return tuple(c)









