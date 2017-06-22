import sys
import math
import numpy as np
from robosims.unity import UnityGame
from PIL import Image, ImageDraw, ImageFont

from util.config import parse_args

def test_unity(argv):
    args = parse_args(argv)
    env = UnityGame(args.conf)
    env.new_episode()

    #move to source
    event = env.take_set_position_action(env.s_x, env.s_y,env.s_z)
    event = env.take_set_rotation_action(0, env.s_r, 0 )
    image_show(event.frame, title="source")

    #move to target
    event = env.take_set_position_action(env.t_x, env.s_y,env.s_z)

    image_show(event.frame, title="target")
    image_show(event.frame_depth, title="target-depth")
    image_show(event.frame_flow, title="source-target-flow")

    input("Press Enter to continue...")
    env.close()


    # rotate to target
    # event = env.take_set_rotation_action(0, env.t_r, 0)


def image_show(a, title=None):
    im = Image.fromarray(a)
    im.show(title=title)
    im.save(title + ".png")


if __name__ == "__main__":
    test_unity(sys.argv[1:])

