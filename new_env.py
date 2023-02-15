import os

import numpy as np
import pybullet as p
import pybullet_data as pd

from tqdm import tqdm
from env import ClutteredPushGrasp
from robot import KinovaRobotiq140
from utilities import YCBModels, Camera
import time
import math


def main(robot):

    ycb_models = YCBModels(
        os.path.join('./data/ycb', '**', 'textured-decmp.obj'),
    )
    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
    camera = None
    env = ClutteredPushGrasp(robot, ycb_models, camera)

    env.reset()
    while True:
        obs, reward, done, info = env.step(
            env.read_debug_parameter(), debug=True)

        # obs, reward, done, info = env.step(
        #     np.array([0., 0.2, 0.5, 0., 0.5, 0., 0.5, 0.]), 'position', debug=True)


if __name__ == "__main__":
    robot = KinovaRobotiq140((0, 0.5, 0), (0, 0, 0), control_mode="position")
    main(robot)
