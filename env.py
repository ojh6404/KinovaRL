import time
import os
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from utilities import Models, Camera
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class FailToReachTargetError(RuntimeError):
    pass


class ClutteredPushGrasp(gym.Env):
    metadata = {'render.modes': ['human']}

    SIMULATION_STEP_DELAY = 1 / 240.
    MAX_EPISODE_LEN = 3000

    def __init__(self, robot, models: Models, camera=None) -> None:
        self.robot = robot
        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.urdfRootPath = pybullet_data.getDataPath()
        self.reset_sim()

        # parameters for gym
        self.num_actions = 8  # 7 for arm, 1 for gripper
        self.action_space = spaces.Box(
            np.array([-1] * self.num_actions), np.array([1] * self.num_actions))
        # ee_pos (3), ee_quat(4), object_pos(3), object_lin_vel(3), arm_pos(7), arm_vel(7), gripper_pos(1)
        # self.observation_space = spaces.Box()

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        if self.robot.control_mode == "end":
            self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
            self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
            self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
            self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
            self.pitchId = p.addUserDebugParameter(
                "pitch", -3.14, 3.14, np.pi/2)
            self.yawId = p.addUserDebugParameter(
                "yaw", -np.pi/2, np.pi/2, np.pi/2)
            self.gripper_opening_length_control = p.addUserDebugParameter(
                "gripper_opening_length", -1., 1., 0.5)
        elif self.robot.control_mode == "position":
            self.j0_input = p.addUserDebugParameter("j0", -1.0, 1.0, 0.)
            self.j1_input = p.addUserDebugParameter("j1", -1.0, 1.0, 0.)
            self.j2_input = p.addUserDebugParameter("j2", -1.0, 1.0, 0.)
            self.j3_input = p.addUserDebugParameter("j3", -1.0, 1.0, 0.)
            self.j4_input = p.addUserDebugParameter("j4", -1.0, 1.0, 0.)
            self.j5_input = p.addUserDebugParameter("j5", -1.0, 1.0, 0.)
            self.j6_input = p.addUserDebugParameter("j6", -1.0, 1.0, 0.)
            self.gripper_input = p.addUserDebugParameter(
                "gripper", -1., 1., 0.)

        self.boxID = p.loadURDF("./urdf/skew-box-button.urdf",
                                [0.0, 0.0, 0.0],
                                # p.getQuaternionFromEuler([0, 1.5706453, 0]),
                                p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True,
                                flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)

        # self.tableID = p.loadURDF(os.path.join(
        #     self.urdfRootPath, "table/table.urdf"), basePosition=[0.0, 0.0, 0.0],
        #     baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        #     useFixedBase=True,
        #     flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION)

        # For calculating the reward
        self.box_opened = False
        self.btn_pressed = False
        self.box_closed = False

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        # if self.vis:
        #     time.sleep(self.SIMULATION_STEP_DELAY)
        #     self.p_bar.update(1)

    def read_debug_parameter(self):
        # read the value of task parameter, not for torque control
        if self.robot.control_mode == "end":
            x = p.readUserDebugParameter(self.xin)
            y = p.readUserDebugParameter(self.yin)
            z = p.readUserDebugParameter(self.zin)
            roll = p.readUserDebugParameter(self.rollId)
            pitch = p.readUserDebugParameter(self.pitchId)
            yaw = p.readUserDebugParameter(self.yawId)
            gripper_opening_length = p.readUserDebugParameter(
                self.gripper_opening_length_control)
            return x, y, z, roll, pitch, yaw, gripper_opening_length
        elif self.robot.control_mode == "position":
            j0 = p.readUserDebugParameter(self.j0_input)
            j1 = p.readUserDebugParameter(self.j1_input)
            j2 = p.readUserDebugParameter(self.j2_input)
            j3 = p.readUserDebugParameter(self.j3_input)
            j4 = p.readUserDebugParameter(self.j4_input)
            j5 = p.readUserDebugParameter(self.j5_input)
            j6 = p.readUserDebugParameter(self.j6_input)
            gripper = p.readUserDebugParameter(self.gripper_input)
            return j0, j1, j2, j3, j4, j5, j6, gripper

    def step(self, actions, debug=False):
        # update vision
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # apply actions
        assert self.robot.control_mode in ('position', 'torque', 'end')
        self.robot.apply_arm_action(actions[:-1], self.robot.control_mode)
        self.robot.move_gripper(actions[-1])

        # for _ in range(120):  # Wait for a few steps

        # step simulator
        self.step_simulation()
        self.step_counter += 1

        obs = self.get_observation()
        reward = self.compute_reward()
        done = self.check_termination()
        info = dict(state=None)
        return obs, reward, done, info

    def compute_reward(self):
        reward = 0
        return reward

    def check_termination(self):
        if self.step_counter > self.MAX_EPISODE_LEN:
            done = True
        else:
            done = False
        return done

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_robot_obs())

        return obs

    def reset_box(self):
        p.setJointMotorControl2(self.boxID, 0, p.POSITION_CONTROL, force=1)
        p.setJointMotorControl2(self.boxID, 1, p.VELOCITY_CONTROL, force=0)

    def reset(self):

        self.reset_sim()
        self.robot.reset()
        self.reset_box()
        self.reset_target()

        return self.get_observation()

    def reset_sim(self):
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setTimeStep(self.SIMULATION_STEP_DELAY)
        p.setGravity(0, 0, -9.81)
        self.planeID = p.loadURDF("plane.urdf")
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def reset_target(self):
        self.target = [1.0, 0.0, 0.0]
        col = [0.5, 1.0, 0.1]
        virtual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                               radius=0.05,
                                               rgbaColor=col)
        target_id = p.createMultiBody(baseMass=0,
                                      baseCollisionShapeIndex=-1,
                                      baseVisualShapeIndex=virtual_shape_id,
                                      basePosition=self.target,
                                      useMaximalCoordinates=True)
        p.resetBasePositionAndOrientation(
            target_id, self.target, np.array([0, 0, 0, 1]))

    def close(self):
        p.disconnect(self.physicsClient)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.05],
                                                          distance=.7,
                                                          yaw=90,
                                                          pitch=-70,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960) / 720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array
