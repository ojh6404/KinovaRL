import pybullet as p
import math
from collections import namedtuple
import numpy as np


class RobotBase(object):
    """
    The base class for robots
    """

    def __init__(self, pos, ori, control_mode="position"):
        """
        Arguments:
            pos: [x y z]
            ori: [r p y]

        Attributes:
            id: Int, the ID of the robot
            eef_id: Int, the ID of the End-Effector
            arm_num_dofs: Int, the number of DoFs of the arm
                i.e., the IK for the EE will consider the first `arm_num_dofs` controllable (non-Fixed) joints
            joints: List, a list of joint info
            controllable_joints: List of Ints, IDs for all controllable joints
            arm_controllable_joints: List of Ints, IDs for all controllable joints on the arm (that is, the first `arm_num_dofs` of controllable joints)

            ---
            For null-space IK
            ---
            arm_lower_limits: List, the lower limits for all controllable joints on the arm
            arm_upper_limits: List
            arm_joint_ranges: List
            arm_rest_poses: List, the rest position for all controllable joints on the arm

            gripper_range: List[Min, Max]
        """
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.control_mode = control_mode

    def load(self):
        self.__init_robot__()
        self.__parse_joint_info__()
        self.__post_load__()
        print(self.joints)

    def step_simulation(self):
        raise RuntimeError(
            '`step_simulation` method of RobotBase Class should be hooked by the environment.')

    def __parse_joint_info__(self):
        numJoints = p.getNumJoints(self.id)
        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'damping', 'friction', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointType = info[2]
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                p.setJointMotorControl2(
                    self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID, jointName, jointType, jointDamping, jointFriction, jointLowerLimit,
                             jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = np.array([
            info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs])
        self.arm_upper_limits = np.array([
            info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs])
        self.arm_joint_ranges = np.array([
            info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs])
        self.arm_torque_limits = np.array([
            info.maxForce for info in self.joints if info.controllable][:self.arm_num_dofs])

        # for scale
        self.joint_pos_mean = (self.arm_upper_limits +
                               self.arm_lower_limits) / 2.
        self.joint_pos_std = (self.arm_upper_limits -
                              self.arm_lower_limits) / 2.

        for joint_id in self.controllable_joints:
            p.enableJointForceTorqueSensor(
                self.id, joint_id, enableSensor=True)

    def __init_robot__(self):
        raise NotImplementedError

    def __post_load__(self):
        pass

    def reset(self):
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        """
        reset to rest poses
        """
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)

        # Wait for a few steps
        for _ in range(10):
            self.step_simulation()

    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def apply_arm_action(self, action, control_method):
        assert control_method in ('position', 'torque', 'end')
        if control_method == 'end':
            x, y, z, roll, pitch, yaw = action
            pos = (x, y, z)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn,
                                                       self.arm_lower_limits, self.arm_upper_limits, self.arm_joint_ranges, self.arm_rest_poses,
                                                       maxNumIterations=20)
            for i, joint_id in enumerate(self.arm_controllable_joints):
                p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                        force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity, positionGain=self.Kp_arm[i], velocityGain=self.Kd_arm[i])
        elif control_method == 'position':
            assert len(action) == self.arm_num_dofs
            joint_poses = self.joint_pos_mean + self.joint_pos_std * action
            for i, joint_id in enumerate(self.arm_controllable_joints):
                p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                        force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity, positionGain=self.Kp_arm[i], velocityGain=self.Kd_arm[i])
        elif control_method == 'torque':
            assert len(action) == self.arm_num_dofs
            torques = self.arm_torque_limits * self.action_scale * action
            torques = np.clip(
                torques, a_min=-1.*self.arm_torque_limits, a_max=self.arm_torque_limits)
            for i, joint_id in enumerate(self.arm_controllable_joints):
                p.setJointMotorControl2(
                    self.id, joint_id, p.TORQUE_CONTROL, force=torques[joint_id])
        else:
            raise NotImplementedError

    def move_gripper(self, open_length):
        raise NotImplementedError

    def get_robot_obs(self):
        positions = []
        velocities = []
        torques = []
        # for joint_id in self.controllable_joints:
        for joint_id in self.arm_controllable_joints:
            pos, vel, _, torque = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
            torques.append(torque)
        ee_pos, ee_ori, _, _, _, _, ee_lin_vel, ee_ang_vel = p.getLinkState(
            self.id, self.eef_id, computeLinkVelocity=True)
        return dict(positions=positions, velocities=velocities, torques=torques, ee_pos=ee_pos, ee_ori=ee_ori, ee_lin_vel=ee_lin_vel, ee_ang_vel=ee_ang_vel)


class KinovaRobotiq140(RobotBase):
    def __init_robot__(self):
        self.eef_id = 7
        self.arm_num_dofs = 7
        # self.arm_rest_poses = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32]
        self.arm_rest_poses = np.array(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035])
        # self.arm_rest_poses = np.array([0., 0., 0., 0., 0., 0., 0.])
        self.id = p.loadURDF('./urdf/gen3_robotiq_2f_140.urdf', self.base_pos, self.base_ori,
                             useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        # self.gripper_range = [0, 0.085]
        self.gripper_range = [-1.0, 1.0]
        # TODO: It's weird to use the same range and the same formula to calculate open_angle as Robotiq85.
        self.actions_scale = 1.0
        self.Kp_arm = np.array(
            [3000., 50000., 3000., 50000., 750., 5000., 100.])
        self.Kd_arm = np.array([2., 0., 0., 0., 0.2, 1., 0.])

        # arm joint limits
        # [-3.141592 -2.41     -3.141592 -2.66     -3.141592 -2.23     -3.141592]
        # [3.141592 2.41     3.141592 2.66     3.141592 2.23     3.141592]

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [
            joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id,
                                   self.id, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            # Note: the mysterious `erp` is of EXTREME importance
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def move_gripper(self, action):
        # action to gripper angle
        gripper_upper_limit = self.joints[self.mimic_parent_id].upperLimit
        gripper_lower_limit = self.joints[self.mimic_parent_id].lowerLimit
        gripper_std = (gripper_upper_limit - gripper_lower_limit) / 2.
        gripper_mean = (gripper_upper_limit + gripper_lower_limit) / 2.
        # open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        open_angle = gripper_mean + gripper_std * action

        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)

    def __post_load__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': -1,
                                'left_inner_knuckle_joint': -1,
                                'right_inner_knuckle_joint': -1,
                                'left_inner_finger_joint': 1,
                                'right_inner_finger_joint': 1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)
