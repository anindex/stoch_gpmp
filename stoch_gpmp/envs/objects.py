
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pybullet as p
from stoch_gpmp.utils import get_assets_path


class BodyCore(ABC):
    def __init__(
        self,
        base_position: Union[np.ndarray, list],
        base_orientation: Union[np.ndarray, list],
        scale: float = 1.0,
        fixed_base: bool = True,
    ) -> None:
        # Store initial position and orientation for resets
        self.init_base_position = base_position
        self.init_base_orientation = base_orientation
        self.fixed_base = fixed_base

        # Stick to base_position/_orientation for conformity with Robot definitions
        self._base_position = base_position
        self._base_orientation = base_orientation

        self.scale = scale

        self.id = None  # To be filled by Bullet client

    @property
    def base_orientation(self):
        return self._base_orientation

    @base_orientation.setter
    def base_orientation(self, values):
        self._base_orientation = values
        if hasattr(self, "client_id"):
            self.client_id.resetBasePositionAndOrientation(
                self.id, self.base_position, self.base_orientation
            )

    @property
    def base_position(self):
        return np.asarray(self._base_position)

    @base_position.setter
    def base_position(self, values):
        self._base_position = values
        if hasattr(self, "client_id"):
            self.client_id.resetBasePositionAndOrientation(
                self.id, self._base_position, self.base_orientation
            )

    def reset(self):
        self.base_orientation = self.init_base_orientation
        self.base_position = self.init_base_position

    @abstractmethod
    def load2client(self, client_id):
        raise NotImplementedError(
            "The method load2client() is not implemented in the abstract class."
        )


class DynamicBodyCore(BodyCore):
    def __init__(
        self,
        base_position: Union[np.ndarray, list] = [0.0, 0.0, 0.0],
        base_orientation: Union[np.ndarray, list] = [0.0, 0.0, 0.0, 1.0],
        base_linear_velocity: Union[np.ndarray, list] = [0.0, 0.0, 0.0],
        base_angular_velocity: Union[np.ndarray, list] = [0.0, 0.0, 0.0],
        scale: float = 1.0,
        fixed_base: bool = True,
    ) -> None:
        super(DynamicBodyCore, self).__init__(
            base_position=base_position,
            base_orientation=base_orientation,
            scale=scale,
            fixed_base=fixed_base,
        )
        # Store initial velocities
        self.init_base_linear_velocity = base_linear_velocity
        self.init_base_angular_velocity = base_angular_velocity

        # Stick to base_velocities for conformity with Robot definitions
        self._base_linear_velocity = base_linear_velocity
        self._base_angular_velocity = base_angular_velocity

    @property
    def base_linear_velocity(self):
        return self._base_linear_velocity

    @base_linear_velocity.setter
    def base_linear_velocity(self, values):
        self._base_linear_velocity = values
        if hasattr(self, "client_id"):
            self.client_id.resetBaseVelocity(
                self.id, self.base_linear_velocity, self.base_angular_velocity
            )

    @property
    def base_angular_velocity(self):
        return self._base_angular_velocity

    @base_angular_velocity.setter
    def base_angular_velocity(self, values):
        self._base_angular_velocity = values
        if hasattr(self, "client_id"):
            self.client_id.resetBaseVelocity(
                self.id, self.base_linear_velocity, self.base_angular_velocity
            )

    def reset(self):
        super().reset()
        self.base_linear_velocity = self.init_base_linear_velocity
        self.base_angular_velocity = self.init_base_angular_velocity

    @abstractmethod
    def load2client(self, client_id):
        raise NotImplementedError(
            "The method load2client() is not implemented in the abstract class."
        )


SPHERE_ROLES = {0: "STATIC_SPHERE", 1: "DYNAMIC_SPHERE"}
SPHERE_COLOR = {
    "STATIC_SPHERE": [1.0, 0.0, 0.0, 1.0],
    "DYNAMIC_SPHERE": [0.5, 0.0, 0.0, 1.0],
}


class Sphere(DynamicBodyCore):
    def __init__(
        self,
        base_position: Union[np.ndarray, list],
        base_linear_velocity: Union[np.ndarray, list],
        scale: float = 0.3,
    ) -> None:
        super(Sphere, self).__init__(
            base_position=base_position,
            base_orientation=[0.0, 0.0, 0.0, 1.0],
            base_linear_velocity=base_linear_velocity,
            base_angular_velocity=[0.0, 0.0, 0.0],
            scale=scale,
            fixed_base=True,
        )
        self._role = None

    @property
    def role(self) -> Union[None, int]:
        return self._role

    @role.setter
    def role(self, value: int) -> None:
        self._role = value

    def reset(self, role: Union[None, int] = None):
        super().reset()
        self.role = role
        if self.role is not None and hasattr(self, "client_id"):
            [
                self.client_id.changeVisualShape(
                    self.id,
                    i,
                    rgbaColor=SPHERE_COLOR[SPHERE_ROLES[self.role]],
                )
                for i in range(-1, 4)
            ]

    def load2client(self, client_id):
        path = (get_assets_path() / 'sphere_simple.urdf').as_posix()
        self.id = client_id.loadURDF(
            path,
            basePosition=self._base_position,
            baseOrientation=self._base_orientation,
            useFixedBase=self.fixed_base,
            globalScaling=self.scale,
        )
        setattr(self, "client_id", client_id)
        return self.id


GAIN_P = 0.1
GAIN_D = 1.0
F_MAX = 250
QLIMITS = 2.96

PANDA = {
    "BASE_POSITION": [0.0, 0.0, 0.05],
    "JOINT_POSITION": [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
}


class Panda(BodyCore):
    def __init__(self, joint_angle: list = None, base_shift: list = [0, 0, 0]) -> None:
        self.initial_joint_positions = (
            joint_angle if joint_angle else PANDA["JOINT_POSITION"]
        )
        self._joint_positions = joint_angle if joint_angle else PANDA["JOINT_POSITION"]
        base_position = list(
            map(lambda x, y: x - y, PANDA["BASE_POSITION"], base_shift)
        )
        super(Panda, self).__init__(
            base_position=base_position, base_orientation=[0.0, 0.0, 0.0, 1.0]
        )
        self.jl_lower = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        self.jl_upper = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]

    @property
    def joint_positions(self):
        return np.asarray(self._joint_positions)

    @joint_positions.setter
    def joint_positions(self, values: list):
        self._joint_positions = values
        if hasattr(self, "client_id"):
            joint_positions = list(self._joint_positions)
            joint_positions.insert(7, 0.0)
            joint_positions.insert(8, 0.0)
            joint_positions.append(0.0)
            [
                self.client_id.resetJointState(
                    self.id, j, targetValue=joint_positions[j]
                )
                for j in range(self.dof)
            ]
        self.resetController()
        self.setTargetPositions(joint_positions)

    def reset(self):
        super(Panda, self).reset()
        self.joint_positions = self.initial_joint_positions
        return self.getJointStates()

    def load2client(self, client_id):
        path = (get_assets_path() / 'franka_description' / 'robots' / 'panda.urdf').as_posix()
        self.id = client_id.loadURDF(
            path,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        setattr(self, "client_id", client_id)

        setattr(self, "dof", self.client_id.getNumJoints(self.id))
        setattr(self, "position_control_gain_p", [GAIN_P for _ in range(self.dof)])
        setattr(self, "position_control_gain_d", [GAIN_D for _ in range(self.dof)])
        setattr(self, "max_joint_torque", [F_MAX for _ in range(self.dof)])

        c = self.client_id.createConstraint(
            self.id,
            8,
            self.id,
            9,
            jointType=p.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.client_id.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

        joint_ids = []
        min_joint_positions = []
        max_joint_positions = []
        target_joint_positions = []
        target_torques = []
        for j in range(self.dof):
            self.client_id.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)
            joint_info = self.client_id.getJointInfo(self.id, j)
            joint_ids.append(j)
            min_joint_positions.append(joint_info[8])
            max_joint_positions.append(joint_info[9])
            target_joint_positions.append(
                (min_joint_positions[j] + max_joint_positions[j]) / 2.0
            )
            target_torques.append(0.0)
        setattr(self, "joint_ids", joint_ids)
        setattr(self, "min_joint_positions", min_joint_positions)
        setattr(self, "max_joint_positions", max_joint_positions)
        setattr(self, "target_joint_positions", target_joint_positions)
        setattr(self, "target_torques", target_torques)
        setattr(self, "pandaEndEffectorIndex", 7)
        self.ee_id = p.getNumJoints(self.id) - 1
        return self.id

    def resetController(self):
        self.client_id.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joint_ids,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for _ in range(self.dof)],
        )

    def setTargetPositions(self, target_joint_positions):
        self.target_joint_positions = self.append(target_joint_positions)
        self.client_id.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.target_joint_positions,
            forces=self.max_joint_torque,
            positionGains=self.position_control_gain_p,
            velocityGains=self.position_control_gain_d,
        )

    def setTargetTorques(self, target_torque):
        self.target_torque = target_torque
        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.joint_ids,
            controlMode=p.TORQUE_CONTROL,
            forces=self.target_torque,
        )

    def getEEPositionAndOrientation(self) -> tuple:
        """Get Position and Orientation in cartesian world-frame.

        :return: ''np.ndarray''
        """
        position, orientation = self.client_id.getLinkState(self.id, self.ee_id)[:2]
        return np.array(position), np.array(orientation)

    def getJointStates(self):
        joint_states = self.client_id.getJointStates(self.id, self.joint_ids)
        joint_position = [x[0] for x in joint_states]
        joint_velocity = [x[1] for x in joint_states]

        self.q = joint_position
        self.dq = joint_velocity

        return (
            joint_position[:7] + joint_position[9:-1],
            joint_velocity[:7] + joint_velocity[9:-1],
        )

    def base_JointStates(self):
        joint_states = self.client_id.getJointStates(self.id, self.joint_ids)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]

        self.q = joint_pos
        self.dq = joint_vel

        return joint_pos[:7] + joint_pos[9:-1], joint_vel[:7] + joint_vel[9:-1]

    def solveInverseDynamics(self, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray):
        return list(self.client_id.calculateInverseDynamics(self.id, pos, vel, acc))

    def solveInverseKinematics(self, pos: np.ndarray, ori: np.ndarray = None):
        if ori is not None:
            return list(self.client_id.calculateInverseKinematics(self.id, self.ee_id, pos, ori, lowerLimits=self.jl_lower, upperLimits=self.jl_upper))
        else:
            return list(self.client_id.calculateInverseKinematics(self.id, self.ee_id, pos, lowerLimits=self.jl_lower, upperLimits=self.jl_upper))

    def append(self, target_pos):
        if len(target_pos) == 9:
            if type(target_pos) == list:
                target_pos.insert(7, 0)
                target_pos.insert(8, 0)
                target_pos.append(0)
                return target_pos
            else:
                target_pos = np.insert(target_pos, 7, 0)
                target_pos = np.insert(target_pos, 8, 0)
                target_pos = np.append(target_pos, 0)

                return target_pos
        return target_pos
