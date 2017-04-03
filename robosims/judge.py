class TaskJudge(object):

    NAVIGATION   = 0
    MANIPULATION = 1
    PROCEDURE    = 2

    TASK_TYPES = [NAVIGATION, MANIPULATION, PROCEDURE]

    COLLISION_PENALTY = 0 # -0.1

    def __init__(self, task_type, task_goal):
        self.task_type = task_type
        self.task_goal = task_goal
        if self.task_type not in TaskJudge.TASK_TYPES:
            raise ValueError('Unknown task type {}'.format(task_type))

    def evaluate(self, event):
        reward = 0   # immediate reward
        done = False # termination flag

        if event.metadata['collided']:
            reward = TaskJudge.COLLISION_PENALTY

        # check success status for navigation tasks
        if self.task_type == TaskJudge.NAVIGATION:
            if self.task_goal in event.metadata['collidedObjects']:
                reward = 1
                done = True

        # check success status for object manipulation
        if self.task_type == TaskJudge.MANIPULATION:
            raise NotImplementedError

        # check success status for procedure tasks
        if self.task_type == TaskJudge.PROCEDURE:
            raise NotImplementedError

        return reward, done