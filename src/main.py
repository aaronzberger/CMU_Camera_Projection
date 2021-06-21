#!/home/aaron/py36/bin/python

# -*- encoding: utf-8 -*-

from numpy.lib.function_base import flip
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
import message_filters
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField, Image
from geometry_msgs.msg import Pose, Point, Vector3
from std_msgs.msg import Header
from transforms import Transforms
from CMU_Mask_R_CNN import msg
from CMU_Camera_Projection.msg import predictions
from vision_msgs.msg import BoundingBox3D
from message_filters import ApproximateTimeSynchronizer
from scipy.spatial.transform import Rotation as R


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

        self.sub_velodyne = message_filters.Subscriber(
            '/velodyne_points', PointCloud2)
        self.sub_predictions = message_filters.Subscriber(
            '/cnn_predictions', msg.predictions)
        ts = ApproximateTimeSynchronizer(
            [self.sub_velodyne, self.sub_predictions], queue_size=20, slop=0.2)
        ts.registerCallback(self.projection_callback)

        self.pub_viz = rospy.Publisher(
            '/projection_viz', Image, queue_size=1)
        self.pub_predictions = rospy.Publisher(
            '/projected_predictions', predictions, queue_size=1)
        self.pub_test = rospy.Publisher(
            'vis_full', Image, queue_size=1)
        self.pub_cloud = rospy.Publisher(
            '/transformed', PointCloud2, queue_size=1)

    def projection_callback(self, data_velo, data_predictions):
        image = cv2.cvtColor(self.cv_bridge.imgmsg_to_cv2(
            data_predictions.source_image, desired_encoding='bgr8'), cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = image.shape

        lidar_ = ros_numpy.numpify(data_velo)
        lidar = np.zeros((lidar_.shape[0], 3))
        lidar[:, 0] = lidar_['x']
        lidar[:, 1] = lidar_['y']
        lidar[:, 2] = lidar_['z']
        # intensities = lidar_['intensity']

        # Final projection from lidar to camera
        proj_lidar_to_cam = self.transforms.intrinsic \
                          @ self.transforms.rect \
                          @ self.transforms.lidar_to_cam

        # Pad with reflectances
        lidar = np.concatenate(
            (lidar, np.ones((lidar.shape[0], 1))), axis=1)

        # Transpose for easier matrix multiplication
        lidar = lidar.transpose()

        # If you wish to publish the transformed pointcloud for testing
        pts_3d = self.transforms.lidar_to_cam @ lidar
        pts_3d = pts_3d.transpose().astype('float32')
        pts_3d[:, 3] = pts_3d[:, 2]
        self.pub_cloud.publish(point_cloud(pts_3d, 'map'))

        # Perform the actual transformation to camera frame
        transformed = proj_lidar_to_cam @ lidar
        transformed[:2, :] /= transformed[2, :]

        # Find indices where the transformed points are within the camera FOV
        inds = np.where((transformed[0, :] < img_width) & (transformed[0, :] >= 0) &
                        (transformed[1, :] < img_height) & (transformed[1, :] >= 0) &
                        (lidar[0, :] > 0)
                        )[0]

        # Get image pixels where there are lidar points
        imgfov_lidar_pixels = transformed[:, inds]
        imgfov_lidar_pixels[:2, :] = np.around(imgfov_lidar_pixels[:2, :]).astype(np.float32)
        # rotation = R.from_euler(seq='x', angles=np.pi / 2)
        # imgfov_lidar_pixels = rotation.apply(imgfov_lidar_pixels.transpose())
        # imgfov_lidar_pixels = imgfov_lidar_pixels.transpose()
        imgfov_lidar_pixels[0, :] -= 25
        for i in range(len(imgfov_lidar_pixels[0, :])):
            imgfov_lidar_pixels[0, i] = max(0, imgfov_lidar_pixels[0, i])

        # Visualization: plot lidar points on the image colored by depth

        img = self.render_on_image(imgfov_lidar_pixels, image)
        ros_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.pub_viz.publish(ros_msg)

        # Get the original lidar points that correspond with the pixel points
        # (usesful for other operations, but will be unused for visualization)
        # lidar_points_in_imgfov = pts_velo.transpose()[inds, :]

        # Calculate depths of the objects
        object_depths = []

        masks = [self.cv_bridge.imgmsg_to_cv2(i) for i in data_predictions.masks]
        # print(max(imgfov_lidar_pixels[0]), max(imgfov_lidar_pixels[1]))
        for mask in masks:
            # print(mask.shape)
            mask_depths = []

            for pixel in imgfov_lidar_pixels.transpose():
                # print(pixel, int(1032 - pixel[1]), int(1384 - pixel[0]))

                if mask[int(1032 - pixel[1] - 1), int(1384 - pixel[0]) - 1]:
                    # print(pixel, (1384 - pixel[0], 1032 - pixel[1]))
                    mask_depths.append(pixel[2])

            if len(mask_depths) == 0:
                object_depths.append(-1)
            else:
                object_depths.append(np.median(mask_depths))

        pub_msg = predictions(header=Header(stamp=data_predictions.header.stamp))
        for box, depth in zip(data_predictions.bboxes, object_depths):
            box_3d = BoundingBox3D()
            box_3d.center = Pose(position=Point(x=depth, y=box.center.x, z=box.center.y))
            box_3d.size = Vector3(x=box.size_x, y=box.size_x, z=box.size_y)
            pub_msg.bboxes.append(box_3d)

        self.pub_predictions.publish(pub_msg)

        img_boxes = np.copy(img)
        img_boxes = cv2.flip(img_boxes, flipCode=-1)
        for box, depth in zip(data_predictions.bboxes, object_depths):
            r_x, r_y = box.size_x / 2, box.size_y / 2
            c_x, c_y = box.center.x, box.center.y

            x = [c_x + r_x, c_x + r_x, c_x - r_x, c_x - r_x]
            y = [c_y + r_y, c_y - r_y, c_y + r_y, c_y - r_y]
            corners = np.empty((4, 2), dtype=np.int)
            corners[:, 0] = x
            corners[:, 1] = y

            text = corners[0][0], corners[0][1] + r_y

            img_boxes = cv2.line(img_boxes, corners[0], corners[1], color=(204, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            img_boxes = cv2.line(img_boxes, corners[0], corners[2], color=(204, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            img_boxes = cv2.line(img_boxes, corners[1], corners[3], color=(204, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            img_boxes = cv2.line(img_boxes, corners[2], corners[3], color=(204, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            img_boxes = cv2.putText(img_boxes, '{}m'.format(int(depth) if depth != -1 else '?'), (corners[0][0], corners[0][1] - int(r_y)), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(204, 255, 255), thickness=2)

        self.pub_test.publish(self.cv_bridge.cv2_to_imgmsg(cv2.flip(img_boxes, flipCode=-1), encoding="bgr8"))

        # print(object_depths)

    def render_on_image(self, lidar_pixels, img):
        # img = cv2.flip(img, flipCode=0)
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        max_depth = np.amax(lidar_pixels[2, :])

        for i in range(lidar_pixels.shape[1]):
            depth = lidar_pixels[2, i]
            color = cmap[int(depth * 255.0 / max_depth), :]
            cv2.circle(img, (int(lidar_pixels[0, i]), int(lidar_pixels[1, i])),
                       2, color=tuple(color), thickness=-1)

        return img


if __name__ == '__main__':
    rospy.init_node('camera_projection', log_level=rospy.INFO)

    rospy.loginfo('Starting camera projection node...')

    camera_projection_node = Camera_Projection_Node()

    rospy.spin()
