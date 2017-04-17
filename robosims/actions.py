from enum import IntEnum

class DiscreteAction(IntEnum):
    Nothing = 0
    Right = 1
    Left = 2
    Up = 3
    Down = 4
    Forward = 5
    Backward = 6
    Clockwise = 7
    AntiClockwise = 8
        
class ActionBuilder(object):
    @staticmethod
    def reset():
        action = dict(actionName='reset=')
        print('reset')
        return action

    @staticmethod
    def put():
        action = dict(actionName  = 'Put')
        return action

    @staticmethod
    def open():
        action = dict(actionName  = 'Open')
        return action

    @staticmethod
    def take():
        action = dict(actionName  = 'Take')
        return action

    @staticmethod
    def close():
        action = dict(actionName  = 'Close')
        return action

    @staticmethod
    def addPositionRotation(x, y, z, rx, ry, rz):
        action = dict(actionName='addPositionRotation')
        p = (x, y, z)
        action['actionVector'] = p
        r = (rx, ry, rz)
        action['actionVector2'] = r
        return action

    @staticmethod
    def addPosition(x, y, z):
        action = dict(actionName='addPosition')
        p = (x, y, z)
        action['actionVector'] = p
        return action

    @staticmethod
    def setPosition(x, y, z):
        action = dict(actionName='setPosition')
        p = (x, y, z)
        action['actionVector'] = p
        return action

    @staticmethod
    def addRotation(rx, ry, rz):
        action = dict(actionName='addRotation')
        r = (rx, ry, rz)
        action['actionVector'] = r
        return action

    @staticmethod
    def setRotation(rx, ry, rz):
        action = dict(actionName='setRotation')
        r = (rx, ry, rz)
        action['actionVector'] = r
        return action

    @staticmethod
    def setVelocity(vx, vy, vz):
        action = dict(actionName='setVelocity')
        v = (vx, vy, vz)
        action['actionVector'] = v
        return action

    @staticmethod
    def setAngularVelocity(wx, wy, wz):
        action = dict(actionName='setAngularVelocity')
        w = (wx, wy, wz)
        action['actionVector']  = w
        return action

    @staticmethod
    def addRelativeForce(fx, fy, fz):
        action = dict(actionName='addRelativeForce')
        f = (fx, fy, fz)
        action['actionVector'] = f
        return action

    @staticmethod
    def addRelativeTorque(tx, ty, tz):
        action = dict(actionName='addRelativeTorque')
        t = (tx, ty, tz)
        action['actionVector'] = t
        return action
