#!/home/aaron/py36/bin/python

# -*- encoding: utf-8 -*-

from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
import message_filters
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Header
from transforms import Transforms
from CMU_Mask_R_CNN import msg
from CMU_Camera_Projection.msg import predictions
from message_filters import ApproximateTimeSynchronizer
import utm
from sensor_msgs.msg import NavSatFix, Imu
import math
from scipy.spatial.transform import Rotation as R
from interpolate_topic import Interpolation_Subscriber
# from CMU_Camera_Projection.msg import trunk


class Camera_Projection_Node:
    def __init__(self):
        self.last_image = Image()
        self.cv_bridge = CvBridge()

        self.transforms = Transforms()
        rospy.loginfo('Waiting to receive camera transforms...')
        self.transforms.get_transforms()
        self.transforms.all_transforms_available.wait()
        rospy.loginfo('Received transforms from camera_info topic. Continuing...')

        self.interpolate_gps = Interpolation_Subscriber(
            '/dji_sdk/gps_position', NavSatFix,
            get_fn=lambda data: [data.latitude, data.longitude])
        self.interpolate_imu = Interpolation_Subscriber(
            '/dji_sdk/imu', Imu,
            get_fn=lambda data: R.from_quat([getattr(
                data.orientation, i) for i in 'xyzw']).as_euler('xyz')[2],
            queue_size=250)

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
        # imgfov_lidar_pixels[0, :] -= 25
        # for i in range(len(imgfov_lidar_pixels[0, :])):
        #     imgfov_lidar_pixels[0, i] = max(0, imgfov_lidar_pixels[0, i])

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

        drone_gps_pos = self.interpolate_gps.get(data_velo.header.stamp)
        drone_utm_pos = utm.from_latlon(*drone_gps_pos)

        # IMU points north, so rotate 90deg to east, since lat/lon points east
        drone_yaw = self.interpolate_imu.get(data_velo.header.stamp) - (math.pi / 2)

        trunks = []

        for object in objects_in_lidar:
            if object is not None:
                obj_gps_offset_x = object[0] * math.cos(-drone_yaw) + object[1] * math.sin(-drone_yaw)
                obj_gps_offset_y = -object[0] * math.sin(-drone_yaw) + object[1] * math.cos(-drone_yaw)

                obj_utm_x = drone_utm_pos[0] + obj_gps_offset_x
                obj_utm_y = drone_utm_pos[1] + obj_gps_offset_y
                object_gps = utm.to_latlon(obj_utm_x, obj_utm_y, zone_number=drone_utm_pos[2], zone_letter=drone_utm_pos[3])

                # TODO: add trunk widths to published message
                # trunks.append(trunk(lat=object_gps[0], lon=object_gps[1]))

                print(drone_gps_pos, object_gps)

        pub_msg = predictions(header=Header(stamp=data_predictions.header.stamp))
        # pub_msg.trunks = trunks
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
