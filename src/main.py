#!/home/aaron/py36/bin/python

# -*- encoding: utf-8 -*-

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
import utm
from sensor_msgs.msg import NavSatFix, Imu
import math
from scipy.spatial.transform import Rotation as R
from interpolate_topic import Interpolation_Subscriber


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

        self.gps_msg_queue = []
        self.gps_time_queue = []
        self.GPS_QUEUE_SIZE = 20

        self.imu_msg_queue = []
        self.imu_time_queue = []
        self.IMU_QUEUE_SIZE = 20

        self.interpolate_gps = Interpolation_Subscriber(
            '/dji_sdk/gps_position', NavSatFix,
            get_fn=lambda data: [data.latitude, data.longitude])
        self.interpolate_imu = Interpolation_Subscriber(
            '/dji_sdk/imu', Imu,
            get_fn=lambda data: R.from_quat([getattr(
                data.orientation, i) for i in 'xyzw']).as_euler('xyz')[2])

        self.sub_velodyne = message_filters.Subscriber(
            '/velodyne_points', PointCloud2)
        self.sub_predictions = message_filters.Subscriber(
            '/cnn_predictions', msg.predictions)
        ts = ApproximateTimeSynchronizer(
            [self.sub_velodyne, self.sub_predictions], queue_size=20, slop=0.2)
        ts.registerCallback(self.projection_callback)

        self.pub_projection = rospy.Publisher(
            '/projection_viz', Image, queue_size=1)
        self.pub_predictions = rospy.Publisher(
            '/projected_predictions', predictions, queue_size=1)
        self.pub_object_depths = rospy.Publisher(
            '/vis_full', Image, queue_size=1)

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

        # Temporary, to fix the extrinsic calibration
        # TODO: Re-calibrate extrinsic transformation and use transforms.py
        imgfov_lidar_pixels[0, :] -= 25
        for i in range(len(imgfov_lidar_pixels[0, :])):
            imgfov_lidar_pixels[0, i] = max(0, imgfov_lidar_pixels[0, i])

        # Visualization: plot lidar points on the image colored by depth
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        max_depth = np.amax(imgfov_lidar_pixels[2, :])

        for i in range(imgfov_lidar_pixels.shape[1]):
            depth = imgfov_lidar_pixels[2, i]
            color = cmap[int(depth * 255.0 / max_depth), :]
            cv2.circle(image, (int(imgfov_lidar_pixels[0, i]), int(imgfov_lidar_pixels[1, i])),
                       2, color=tuple(color), thickness=-1)

        ros_msg = self.cv_bridge.cv2_to_imgmsg(image, encoding="bgr8")
        self.pub_projection.publish(ros_msg)

        # Calculate depths of the objects
        objects_in_lidar = []
        lidar_points_in_imgfov = lidar.transpose()[inds, :]

        masks = [self.cv_bridge.imgmsg_to_cv2(i) for i in data_predictions.masks]
        for mask in masks:
            mask_x = []
            mask_y = []
            mask_z = []

            for index, pixel in enumerate(imgfov_lidar_pixels.transpose()):
                if mask[int(1032 - pixel[1] - 1), int(1384 - pixel[0]) - 1]:
                    x, y, z = lidar_points_in_imgfov[index][:3]

                    mask_x.append(x)
                    mask_y.append(y)
                    mask_z.append(z)

                    color = cmap[int(depth * 255.0 / max_depth), :]
                    corners = ((int(pixel[0] + 4), int(pixel[1] + 4)), (int(pixel[0] - 4), int(pixel[1] - 4)))
                    cv2.rectangle(image, corners[0], corners[1], color=tuple(color), thickness=-1)

            if len(mask_x) == 0:
                objects_in_lidar.append(None)
            else:
                objects_in_lidar.append([np.median(mask_x), np.mean(mask_y), np.mean(mask_z)])

        gps_position = self.interpolate_gps.get(data_velo.header.stamp)
        utm_pos = utm.from_latlon(*gps_position)

        yaw_offset = self.interpolate_imu.get(data_velo.header.stamp) - (math.pi / 2)

        for box, object in zip(data_predictions.bboxes, objects_in_lidar):
            if object is not None:
                print(box.center.x, box.center.y, object[:3])
                new_x = object[0] * math.cos(-yaw_offset) + object[1] * math.sin(-yaw_offset)
                new_y = -object[0] * math.sin(-yaw_offset) + object[1] * math.cos(-yaw_offset)

                # print(new_x, new_y)
                utm_x = utm_pos[0] + new_x
                utm_y = utm_pos[1] + new_y
                new_gps = utm.to_latlon(utm_x, utm_y, zone_number=utm_pos[2], zone_letter=utm_pos[3])
                print(gps_position, new_gps)

        # Publish the predictions message
        pub_msg = predictions(header=Header(stamp=data_predictions.header.stamp))
        for box, depth in zip(data_predictions.bboxes, objects_in_lidar):
            box_3d = BoundingBox3D()
            box_3d.center = Pose(position=Point(x=depth, y=box.center.x, z=box.center.y))
            box_3d.size = Vector3(x=box.size_x, y=box.size_x, z=box.size_y)
            pub_msg.bboxes.append(box_3d)

        self.pub_predictions.publish(pub_msg)

        # Visualization: plot lidar points on image colored by depth and draw
        # predicted bounding boxes and estimated depth (distance from drone)
        img_boxes = cv2.flip(np.copy(image), flipCode=-1)
        for box, object in zip(data_predictions.bboxes, objects_in_lidar):
            r_x, r_y = box.size_x / 2, box.size_y / 2
            c_x, c_y = box.center.x, box.center.y
            x = [c_x + r_x, c_x + r_x, c_x - r_x, c_x - r_x]
            y = [c_y + r_y, c_y - r_y, c_y + r_y, c_y - r_y]

            corners = np.empty((4, 2), dtype=np.int)
            corners[:, 0] = x
            corners[:, 1] = y

            if object is None:
                thickness = 2
            else:
                thickness = 3

            img_boxes = cv2.line(img_boxes, corners[0], corners[1], color=(204, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
            img_boxes = cv2.line(img_boxes, corners[0], corners[2], color=(204, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
            img_boxes = cv2.line(img_boxes, corners[1], corners[3], color=(204, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
            img_boxes = cv2.line(img_boxes, corners[2], corners[3], color=(204, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
            img_boxes = cv2.putText(img_boxes, '{}m'.format(int(object[0]) if object is not None else '?'), (corners[0][0], corners[0][1] - int(r_y)), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(204, 255, 255), thickness=2)

        self.pub_object_depths.publish(self.cv_bridge.cv2_to_imgmsg(cv2.flip(img_boxes, flipCode=-1), encoding="bgr8"))


if __name__ == '__main__':
    rospy.init_node('camera_projection', log_level=rospy.INFO)

    rospy.loginfo('Starting camera projection node...')

    camera_projection_node = Camera_Projection_Node()

    rospy.spin()
