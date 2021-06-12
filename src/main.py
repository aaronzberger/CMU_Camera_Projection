#!/home/aaron/py36/bin/python

# -*- encoding: utf-8 -*-

from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
from transforms import Transforms


def point_cloud(points, parent_frame):
    '''
    Create a PointCloud2 message.

    Parameters:
        points: Nx4 array of xyz positions (m) and reflectances (0-1)
        parent_frame: frame in which the point cloud is defined

    Returns:
        sensor_msgs/PointCloud2 message
    '''
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate(['x', 'y', 'z', 'intensity'])]

    header = Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 4),
        row_step=(itemsize * 4 * points.shape[0]),
        data=data
    )


class Camera_Projection_Node:
    def __init__(self):
        self.last_image = Image()
        self.cv_bridge = CvBridge()

        self.transforms = Transforms()
        rospy.loginfo('Waiting to receive camera transforms...')
        self.transforms.get_transforms()
        self.transforms.all_transforms_available.wait()
        rospy.loginfo('Received transforms from camera_info topic. Continuing...')

        self.sub_velodyne = rospy.Subscriber(
            '/velodyne_points', PointCloud2, self.velodyne_callback)
        self.sub_cam = rospy.Subscriber(
            '/mapping/left/image_raw', Image, self.image_callback)
        self.pub_viz = rospy.Publisher(
            '/projection_viz', Image, queue_size=1)
        self.pub_cloud = rospy.Publisher(
            '/transformed', PointCloud2, queue_size=1)

    def image_callback(self, data):
        self.last_image = data

    def velodyne_callback(self, data):
        image = cv2.cvtColor(self.cv_bridge.imgmsg_to_cv2(
            self.last_image, desired_encoding='bgr8'), cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = image.shape

        lidar_ = ros_numpy.numpify(data)
        lidar = np.zeros((lidar_.shape[0], 3))
        lidar[:, 0] = lidar_['x']
        lidar[:, 1] = lidar_['y']
        lidar[:, 2] = lidar_['z']
        # intensities = lidar_['intensity']

        self.render_lidar_on_image(lidar, image, img_width, img_height)

    def render_lidar_on_image(self, pts_velo, img, img_width, img_height):
        # Final projection from lidar to camera
        proj_lidar_to_cam = self.transforms.intrinsic \
                          @ self.transforms.rect \
                          @ self.transforms.lidar_to_cam

        # Pad with reflectances
        pts_velo = np.concatenate(
            (pts_velo, np.ones((pts_velo.shape[0], 1))), axis=1)

        # Transpose for easier matrix multiplication
        pts_velo = pts_velo.transpose()

        # If you wish to publish the transformed pointcloud for testing
        pts_3d = self.transforms.lidar_to_cam @ pts_velo
        pts_3d = pts_3d.transpose().astype('float32')
        pts_3d[:, 3] = pts_3d[:, 2]
        self.pub_cloud.publish(point_cloud(pts_3d, 'map'))

        # Perform the actual transformation to camera frame
        transformed = proj_lidar_to_cam @ pts_velo
        transformed[:2, :] /= transformed[2, :]

        # Find indices where the transformed points are within the camera FOV
        inds = np.where((transformed[0, :] < img_width) & (transformed[0, :] >= 0) &
                        (transformed[1, :] < img_height) & (transformed[1, :] >= 0) &
                        (pts_velo[0, :] > 0)
                        )[0]

        # Get image pixels where there are lidar points
        imgfov_lidar_pixels = transformed[:, inds]

        # Get the original lidar points that correspond with the pixel points
        # (usesful for other operations, but will be unused for visualization)
        # lidar_points_in_imgfov = pts_velo.transpose()[inds, :]

        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        max_depth = np.amax(imgfov_lidar_pixels[2, :])

        for i in range(imgfov_lidar_pixels.shape[1]):
            depth = imgfov_lidar_pixels[2, i]
            color = cmap[int(depth * 255.0 / max_depth), :]
            cv2.circle(img, (int(np.round(imgfov_lidar_pixels[0, i])),
                             int(np.round(imgfov_lidar_pixels[1, i]))),
                       2, color=tuple(color), thickness=-1)

        ros_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="bgr8")

        self.pub_viz.publish(ros_msg)


if __name__ == '__main__':
    rospy.init_node('camera_projection', log_level=rospy.INFO)

    rospy.loginfo('Starting camera projection node...')

    camera_projection_node = Camera_Projection_Node()

    rospy.spin()
