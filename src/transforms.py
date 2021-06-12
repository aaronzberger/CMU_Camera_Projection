import rospy

from scipy.spatial.transform import Rotation as R
import numpy as np
from sensor_msgs.msg import CameraInfo
import threading


class Transforms:
    def __init__(self):
        lidar_to_imu = [[1,  0,  0, -0.011356],
                        [0, -1,  0, -0.002352],
                        [0,  0, -1, -0.08105],
                        [0,  0,  0, 1]]

        # Transform imu to left camera (input XYZW)
        rot_imu_to_left = R.from_quat([0.503096384820381, -0.49276446989981,
                                       0.500896743794452, -0.501684436262218])
        imu_to_left = np.eye(4)
        imu_to_left[:3, :3] = rot_imu_to_left.as_matrix()
        imu_to_left[:3, 3] = [0.0870551272380779, -0.107604788194452, 0.0180391607070435]

        # Combine above transforms to get lidar to left camera transform
        self.lidar_to_cam = np.matmul(np.linalg.inv(imu_to_left), lidar_to_imu)

        self.all_transforms_available = threading.Event()

    def get_transforms(self, timeout=None):
        '''
        Get the intrinsic and rectification transforms

        Parameters:
            timeout (float): seconds to wait for the camera_info message
        '''
        camera_info_msg = rospy.wait_for_message(
            '/mapping/left/camera_info', CameraInfo,
            timeout=timeout)
        np.set_printoptions(suppress=True)

        # rot_rect = np.array(camera_info_msg.R).reshape(3, 3)
        # self.rect = np.eye(4)
        # self.rect[:3, :3] = rot_rect

        # self.intrinsic = np.array(camera_info_msg.P).reshape(3, 4)

        # For some reason, only the below transforms are working. I don't know where I got them.

        # Rectification matrix (3x3 rotation inside 4x4 identity)
        rot_rect = np.array([[0.99996213,   -0.00211377, -0.00844186],
                            [0.00213563,  0.99999439, 0.00258106],
                            [0.00843636, -0.00259899, 0.99996104]])
        self.rect = np.eye(4)
        self.rect[:3, :3] = rot_rect

        # Intrinsic camera parameters (always 3x4 matrix)
        self.intrinsic = np.array([[1431.6832, 0.0, 704.606, 0.0],
                                   [0.0, 1431.6832,  545.11274, 0.0],
                                   [0.0,       0.0,        1.0, 0.0]])

        self.all_transforms_available.set()
