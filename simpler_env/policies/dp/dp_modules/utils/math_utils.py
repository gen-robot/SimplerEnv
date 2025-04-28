import numpy as np
from transforms3d.euler import quat2euler as quat2eulertuple
from transforms3d.euler import euler2quat as eulertuple2quat
from transforms3d.euler import euler2mat as eulertuple2mat
from transforms3d.euler import mat2euler as mat2eulertuple


def wrap_to_pi(x):
    x = x % (2 * np.pi)
    x = x - 2 * np.pi * (x > np.pi)
    return x


def quat2euler(q, *args, **kwargs):
    return np.array(quat2eulertuple(q, *args, **kwargs))


def euler2quat(x: np.ndarray):
    return eulertuple2quat(*(x.tolist()))


def euler2mat(x: np.ndarray):
    return eulertuple2mat(*(x.tolist()))


def mat2euler(x: np.ndarray):
    return np.array(mat2eulertuple(x))


def get_pose_from_rot_pos(mat: np.ndarray, pos: np.ndarray):
    return np.concatenate(
        [
            np.concatenate([mat, pos.reshape(3, 1)], axis=-1),
            np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4),
        ],
        axis=0,
    )


def rot6d2mat(x: np.ndarray):
    x = x.reshape(3, 2)
    a1, a2 = x[:, 0], x[:, 1]
    b1 = a1 / np.linalg.norm(a1)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / np.linalg.norm(a2)
    b3 = np.cross(b1, b2)
    mat = np.stack([b1, b2, b3], axis=-1)
    return mat
