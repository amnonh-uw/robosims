import random
from robosims.actions import ActionBuilder

# markov process model
class Model(object):
    def __init__(self):
        self.state = self.start_state
        self.forward_speed_x = 0
        self.forward_speed_z = 0
        self.rotation_speed = 0

    def next_action(self, event):
        # The markov process model does not look at the state from unity
        return self.state()

    def reset(self):
        self.state = self.start_sate

    def start_state(self):
        self.forward_speed_x = random.random() * 2 - 1
        self.forward_speed_z = random.random() * 2 - 1
        self.state = self.forward_state
        return ActionBuilder.setVelocity(self.forward_speed_x, 0, self.forward_speed_z)

    def forward_state(self):
        if random.random() > 0.9:
            self.state = self.rotate_state
            self.rotation_speed = random.random() * 2 - 1
            return ActionBuilder.setAngularVelocity(self.rotation_speed, 0, 0)

        return ActionBuilder.setVelocity(self.forward_speed_x, 0, self.forward_speed_z)

    def rotate_state(self):
        if random.random() > 0.9:
            self.state = self.forward_state
            self.forward_speed_x = random.random() * 2 - 1
            self.forward_speed_z = random.random() * 2 - 1
            return ActionBuilder.setVelocity(self.forward_speed_x, 0, self.forward_speed_z)

        return ActionBuilder.setAngularVelocity(self.rotation_speed, 0, 0)
