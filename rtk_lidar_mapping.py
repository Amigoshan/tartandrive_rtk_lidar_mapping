import numpy as np
import os
import cv2
import rerun as rr
from point_undistort import undistort_points

class PinholeCameraModel:
    def __init__(self, img_width, img_height, focalx, focaly, pu, pv):
        self.img_width = img_width
        self.img_height = img_height
        self.focalx = focalx
        self.focaly = focaly
        self.pu = pu
        self.pv = pv
        self.intrinsic_matrix = np.array([[focalx, 0, pu],
                                          [0, focaly, pv],
                                          [0, 0, 1]])
        
    def project_points(self, points_3d):
        '''
        points_3d: (N, 3) in camera frame
        returns: a depth image of shape (img_height, img_width) where each pixel value is the depth of the point projected onto that pixel, or 0 if no point projects there.
        this calculation is fast to compute for a large number of points, as it uses numpy vectorized operations.
        '''
        # filter points that are behind the camera
        in_front = points_3d[:, 2] > 0
        points_3d = points_3d[in_front]
        # project points to 2D
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]
        u = (self.focalx * x / z) + self.pu
        v = (self.focaly * y / z) + self.pv
        # round to nearest pixel and convert to int
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        # create depth image
        depth_image = np.full((self.img_height, self.img_width), np.inf, dtype=np.float32)
        # filter points that project within the image bounds
        valid = (u >= 0) & (u < self.img_width) & (v >= 0) & (v < self.img_height)
        u = u[valid]
        v = v[valid]
        z = z[valid]

        # Vectorized z-buffer update: keep nearest depth per projected pixel.
        np.minimum.at(depth_image, (v, u), z)
        depth_image[~np.isfinite(depth_image)] = 0.0
        return depth_image

    def colorize_points(self, points_3d, image, stereo_dist = None, threshold=0.2): 
        '''
        points_3d: (N, 3) in camera frame
        image: (H, W, 3) RGB image
        stereo_dist: (H, W) stereo distance image, optional. Note this is not stereo_depth, but the distance from the camera to the point in 3D space, which is different from depth. If provided, points that are not consistent with stereo_dist will be filtered out.
        returns: a colorized point cloud of shape (M, 6) where each point has (x, y, z, r, g, b), points not in view are disgarded, and M <= N
        '''
        valid_mask = points_3d[:, 2] > 0
        points_3d = points_3d[valid_mask]
        # project points to 2D
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]
        u = (self.focalx * x / z) + self.pu
        v = (self.focaly * y / z) + self.pv
        # round to nearest pixel and convert to int
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        # filter points that project within the image bounds
        valid = (u >= 0) & (u < self.img_width) & (v >= 0) & (v < self.img_height) 
        u = u[valid]
        v = v[valid]
        z = z[valid]
        points_3d_valid = points_3d[valid]
        valid_mask[valid_mask.nonzero()[0][~valid]] = False  # Update the valid mask to reflect the projection filtering

        if stereo_dist is not None:
            # filter points that are consistent with stereo_dist
            stereo_depth_at_points = stereo_dist[v, u]
            depth_diff = np.abs(stereo_depth_at_points - z)
            depth_accurate_mask = depth_diff < threshold
            u = u[depth_accurate_mask]
            v = v[depth_accurate_mask]
            points_3d_valid = points_3d_valid[depth_accurate_mask]

            valid_mask[valid_mask.nonzero()[0][~depth_accurate_mask]] = False  # Update the valid mask to reflect the depth accuracy filtering

        # get colors from image
        colors = image[v, u]  # shape (N_valid, 3)
        
        # concatenate points and colors
        colorized_points = np.hstack((points_3d_valid, colors))
        return colorized_points, valid_mask

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    R = np.array([[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                  [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                  [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]])
    return R

def euler_to_rotation_matrix(roll, pitch, yaw, first_axis='yaw'):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    if first_axis == 'yaw':
        R = R_z @ R_y @ R_x
    elif first_axis == 'pitch':
        R = R_y @ R_x @ R_z
    elif first_axis == 'roll':
        R = R_x @ R_z @ R_y

    # R = R_z @ R_y @ R_x  # This line is redundant and should be removed
    return R

def pointcloud_transform(points, transform):
    '''
    points: (N, 3)
    transform: 4x4 homogeneous transformation matrix
    returns: transformed points of shape (N, 3)
    '''
    # convert to homogeneous coordinates
    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))
    # apply transformation
    transformed_homogeneous = homogeneous_points @ transform.T
    # convert back to Cartesian coordinates
    transformed_points = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3:]
    return transformed_points

def pose7_to_matrix(pose7):
    # convert to 4x4 homogeneous transformation matrix
    tx, ty, tz, qx, qy, qz, qw = pose7
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T

def matrix_to_pose7(T): 
    # convert 4x4 homogeneous transformation matrix to 7D pose
    tx, ty, tz = T[:3, 3]
    R = T[:3, :3]
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return np.array([tx, ty, tz, qx, qy, qz, qw])


def _log_transform_compat(entity_path, T):
    """Log a 4x4 transform in both new and old Rerun APIs."""
    if hasattr(rr, "log_transform3d"):
        rr.log_transform3d(entity_path, T)
        return

    R = T[:3, :3]
    t = T[:3, 3]
    rr.log(entity_path, rr.Transform3D(translation=t, mat3x3=R))

def _log_rgb_axes(entity_path, axis_length=0.5):
    """Draw local XYZ frame axes: X=red, Y=green, Z=blue."""
    rr.log(
        f"{entity_path}/axes/x",
        rr.LineStrips3D(
            [[[0.0, 0.0, 0.0], [axis_length, 0.0, 0.0]]],
            colors=[[255, 0, 0]],
            radii=[0.01],
        ),
    )
    rr.log(
        f"{entity_path}/axes/y",
        rr.LineStrips3D(
            [[[0.0, 0.0, 0.0], [0.0, axis_length, 0.0]]],
            colors=[[0, 255, 0]],
            radii=[0.01],
        ),
    )
    rr.log(
        f"{entity_path}/axes/z",
        rr.LineStrips3D(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, axis_length]]],
            colors=[[0, 0, 255]],
            radii=[0.01],
        ),
    )

def rerun_plot_axes(T, init=True):
    if init:
        rr.init('extrinsics_test', spawn=True)
    # visualize the origin frame using 3 axis markers
    _log_transform_compat("origin", np.eye(4))
    _log_rgb_axes("origin")
    _log_transform_compat("lidar_to_camera", T)
    _log_rgb_axes("lidar_to_camera")

class LidarLoader:
    '''
    Return points in the lidar frame, optionally at time t
    '''
    def __init__(self, traj_folder):
        self.traj_folder = traj_folder
        self.lidar_folder = None
        self.velodyne_timestamps = None

    def load_pointcloud(self, index):
        lidar_path = os.path.join(self.lidar_folder, f'{index:08d}.npy')
        pointcloud = np.load(lidar_path)[:, :3] # (N, 3)
        return pointcloud

    def find_closest_velodyne_index(self, timestamp):
        idx = np.argmin(np.abs(self.velodyne_timestamps - timestamp))
        return idx, self.velodyne_timestamps[idx]

def interpolate_pose_based_on_timestamps(timestamps, poses, target_timestamps):
    """
    Interpolate poses based on timestamps.

    :param timestamps: Original timestamps (N,)
    :param poses: Original poses (N, 4, 4)
    :param target_timestamps: Target timestamps to interpolate to (M,)
    :return: Interpolated poses (M, 4, 4)
    """
    interpolated_poses = []
    for t in target_timestamps:
        if t <= timestamps[0]:
            interpolated_poses.append(poses[0])
        elif t >= timestamps[-1]:
            interpolated_poses.append(poses[-1])
        else:
            idx = np.searchsorted(timestamps, t) - 1
            t0, t1 = timestamps[idx], timestamps[idx + 1]
            pose0, pose1 = poses[idx], poses[idx + 1]
            alpha = (t - t0) / (t1 - t0)
            # Interpolate SE(3): LERP translation and SLERP rotation.
            p0, p1 = pose0[:3, 3], pose1[:3, 3]
            p = (1.0 - alpha) * p0 + alpha * p1

            q0 = matrix_to_pose7(pose0)[3:7]
            q1 = matrix_to_pose7(pose1)[3:7]
            q0 = q0 / np.linalg.norm(q0)
            q1 = q1 / np.linalg.norm(q1)

            # Keep shortest arc by flipping quaternion hemisphere when needed.
            if np.dot(q0, q1) < 0.0:
                q1 = -q1

            dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
            if dot > 0.9995:
                q = (1.0 - alpha) * q0 + alpha * q1
                q = q / np.linalg.norm(q)
            else:
                theta = np.arccos(dot)
                sin_theta = np.sin(theta)
                w0 = np.sin((1.0 - alpha) * theta) / sin_theta
                w1 = np.sin(alpha * theta) / sin_theta
                q = w0 * q0 + w1 * q1

            interpolated_pose = pose7_to_matrix(np.concatenate([p, q]))
            interpolated_poses.append(interpolated_pose)
    return np.array(interpolated_poses)


class VelodyneRTKLoader():
    '''
    Load velodyne pointclouds and RTK odometry, and transform pointclouds to local frame using odometry.
    Undistort the pointclouds using the relative motion between the start and end of the scan.
    '''
    def __init__(self, traj_folder, rtk_lidar_extrinsics):
        '''
        rtk_lidar_extrinsics: 4x4 transformation matrix from rtk frame to velodyne frame
        '''
        self.lidar_folder = os.path.join(traj_folder, 'sensors/velodyne_1')
        self.velodyne_timestamps = np.loadtxt(os.path.join(self.lidar_folder, 'timestamps.txt'))

        rtk_file = f'{traj_folder}/sensors/gq7_rtk_odometry/interp_data.txt'
        rtk_timestamps_file = f'{traj_folder}/sensors/gq7_rtk_odometry/interp_timestamps.txt'
        rtk_data = np.loadtxt(rtk_file)
        self.rtk_timestamps = np.loadtxt(rtk_timestamps_file)

        lidar_rtk_extrinsics = np.linalg.inv(rtk_lidar_extrinsics) # transform from rtk frame to velodyne frame
        rtk_poses = np.array([pose7_to_matrix(pose) for pose in rtk_data[:, :7]])
        rtk_pose0_inv = np.linalg.inv(rtk_poses[0])
        self.rtk_poses_transformed = np.array([rtk_lidar_extrinsics @ rtk_pose0_inv @ pose @ lidar_rtk_extrinsics for pose in rtk_poses])

    def load_pointcloud_by_time(self, timestamp, timeoffset = 0.0, accumulate_num=1, frame = 'velodyne_at_timestamp'):
        '''
        find the closest velodyne frame
        accumulate multiple previous frames 
        undistort the points using the relative motion between the start and end of the scan
        transform the points to the frame at the target timestamp
        frame: 'velodyne_init' or 'velodyne_at_timestamp'
        '''

        pointclouds_undistorted = []
        index, closest_timestamp = self.find_closest_velodyne_index(timestamp)
        # for debug
        # print(f"Requested timestamp: {timestamp}, closest velodyne timestamp: {closest_timestamp}, index: {index}")

        for k in range(accumulate_num):
            if index - k >= 0:
                lidar_file = os.path.join(self.lidar_folder, f'{index - k:08d}.npy')
                pointcloud_with_time = np.load(lidar_file) # (N, 4) [x, y, z, t]

                point_distances = np.linalg.norm(pointcloud_with_time[:, :3], axis=1)
                valid_points = np.logical_and(point_distances > 1.9, point_distances < 200.0)  
                pointcloud_with_time = pointcloud_with_time[valid_points]

                endtime_points = self.velodyne_timestamps[index - k]
                starttime_points = self.velodyne_timestamps[index - k] - 0.1
                pose_start, pose_end = interpolate_pose_based_on_timestamps(self.rtk_timestamps, self.rtk_poses_transformed, np.array([starttime_points, endtime_points ]))
                delta_pose = np.linalg.inv(pose_end) @ pose_start

                undistorted_points = undistort_points(pointcloud_with_time, delta_pose) # velodyne frame at endtime_points

                pointcloud_transformed_undistorted = pointcloud_transform(undistorted_points[:, :3], pose_end) # velodyne frame to velodyne_init frame
                pointclouds_undistorted.append(pointcloud_transformed_undistorted)

        pointclouds_undistorted = np.vstack(pointclouds_undistorted)

        if frame == 'velodyne_init':
            return pointclouds_undistorted, closest_timestamp

        elif frame == 'velodyne_at_timestamp':
        
            target_pose_time = timestamp + timeoffset
            target_pose = interpolate_pose_based_on_timestamps(self.rtk_timestamps, self.rtk_poses_transformed, np.array([target_pose_time]))[0]
            # transform back to local frame (the frame wrt the timestamp + timeoffset)
            pointcloud_local = pointcloud_transform(pointclouds_undistorted, np.linalg.inv(target_pose))

            return pointcloud_local, target_pose_time

        else: 
            raise ValueError(f"Unknown frame type: {frame}")

    def load_pointcloud_and_colorize(self, index, image, rgb_timestamp, camera_model, extrinsics, stereo_dist = None, timeoffset = 0.0):
        '''
        load pointcloud and colorize it using the closest rgb image
        '''
        lidar_file = os.path.join(self.lidar_folder, f'{index:08d}.npy')
        pointcloud_with_time = np.load(lidar_file) # (N, 4) [x, y, z, t]

        point_distances = np.linalg.norm(pointcloud_with_time[:, :3], axis=1)
        valid_points = np.logical_and(point_distances > 1.9, point_distances < 200.0)  
        pointcloud_with_time = pointcloud_with_time[valid_points]

        endtime_points = self.velodyne_timestamps[index]
        starttime_points = self.velodyne_timestamps[index] - 0.1
        pose_start, pose_end = interpolate_pose_based_on_timestamps(self.rtk_timestamps, self.rtk_poses_transformed, np.array([starttime_points, endtime_points ]))
        delta_pose = np.linalg.inv(pose_end) @ pose_start

        undistorted_points = undistort_points(pointcloud_with_time, delta_pose) # in velodyne frame at endtime_points
        undistorted_points = pointcloud_transform(undistorted_points[:, :3], pose_end) # transform to velodyne_init frame

        image_frame_pose = interpolate_pose_based_on_timestamps(self.rtk_timestamps, self.rtk_poses_transformed, np.array([rgb_timestamp + timeoffset]))[0]
        pointcloud_transformed_to_camera = pointcloud_transform(undistorted_points[:, :3], np.linalg.inv(image_frame_pose))
        pointcloud_transformed_to_image = pointcloud_transform(pointcloud_transformed_to_camera, extrinsics)

        colorized_pointcloud, valid_mask  = camera_model.colorize_points(pointcloud_transformed_to_image, image, stereo_dist = stereo_dist, threshold=0.2)

        valid_pointcloud_init_frame = undistorted_points[valid_mask, :3]
        colorized_pointcloud_init_frame = np.hstack((valid_pointcloud_init_frame, colorized_pointcloud[:, 3:]))

        return colorized_pointcloud_init_frame, pose_end


def accumulate_pointclouds_using_rtk(traj_folder, rtk_lidar_extrinsics, lidar_camera_extrinsics, timeoffset = 0.0, framelist = None, startframe=0, endframe=None, colorize=True):
    '''
    Accumulate pointclouds for the entire sequence using RTK odometry
    Colorize the pointclouds using the closest rgb image
    '''
    lidar_loader = VelodyneRTKLoader(traj_folder, rtk_lidar_extrinsics)
    rgb_folder = 'sensors/multisense_left_rect_color'
    camera_timestamps = np.loadtxt(os.path.join(traj_folder, rgb_folder, 'timestamps.txt'))
    img_w, img_h = 1024, 544
    focalx = 477.6049499511719
    focaly = 477.6049499511719
    pu = 499.5
    pv = 252.0
    camera_model = PinholeCameraModel(img_w, img_h, focalx, focaly, pu, pv)

    framenum = camera_timestamps.shape[0]
    endframe = framenum if endframe is None else min(endframe, framenum)
    if colorize:
        rr.init('accumulated_pointcloud', spawn=True)
    acc_pointcloud = []
    indexlist = framelist if framelist is not None else range(startframe, endframe, 1)
    for index in indexlist:
        rgb_path = os.path.join(traj_folder, rgb_folder, f'{index:08d}.png')
        rgb_image = cv2.imread(rgb_path)
        rgb_image = rgb_image[:, :, ::-1]  # convert BGR to RGB
        rgb_time = camera_timestamps[index]

        if colorize:
            colorized_pointcloud, pose = lidar_loader.load_pointcloud_and_colorize(index, rgb_image, rgb_time, camera_model, lidar_camera_extrinsics, stereo_dist=None, timeoffset=timeoffset)

            # visualize the pointcloud in rerun
            rr_vis_color = colorized_pointcloud[:, 3:] / 255.0  # normalize to [0, 1] for rerun
            rr.log(f"pointcloud_{index}", rr.Points3D(colorized_pointcloud[:, :3], colors=rr_vis_color, radii=0.01))
            # visualize camera pose
            _log_transform_compat(f"lidar_{index}", pose)
            _log_rgb_axes(f"lidar_{index}")

            acc_pointcloud.append(colorized_pointcloud)

        else: 
            pointcloud, _ = lidar_loader.load_pointcloud_by_time(rgb_time, timeoffset=timeoffset, accumulate_num=1, frame='velodyne_init')
            acc_pointcloud.append(pointcloud)

    # import ipdb;ipdb.set_trace()
    return np.vstack(acc_pointcloud)


if __name__ == '__main__':

    # from rtk to lidar frame
    rtk_lidar_extrinsics = np.array([[ 0.9994685,  -0.03138012, -0.00883182, -0.23],
                                    [ 0.03140956,  0.99950143,  0.00321482, 0.02],
                                    [ 0.00872654, -0.00349052,  0.99995583, 0.13],
                                    [ 0.,          0.,          0.,          1.]])
    # from lidar to camera frame
    lidar_camera_extrinsics = np.array([
        [0.007624, -0.999948,  0.006789,  0.1089983],
        [-0.220747, -0.008305, -0.975296, -0.15198574],
        [0.975301,  0.005937, -0.220799,  0.05266748],
        [0.0,        0.0,        0.0,        1.0]
    ])

    traj_folder = '/home/wenshan/tmp/offroad_test/2026-07-10-15-26-09_00_warehouse_calib_1_so_postproc'
    acc_points = accumulate_pointclouds_using_rtk(traj_folder, 
                                     rtk_lidar_extrinsics= rtk_lidar_extrinsics, 
                                     lidar_camera_extrinsics=lidar_camera_extrinsics, 
                                     startframe = 100, endframe=1000) #'/home/wenshan/tmp/offroad_test/2026-07-10-15-26-09_00_warehouse_calib_1_so_postproc')
