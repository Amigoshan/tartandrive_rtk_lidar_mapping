import numpy as np


_EPS = 1e-12


def _normalize_quaternion_xyzw(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < _EPS:
        raise ValueError("Quaternion norm is near zero")
    return q / n


def _quat_xyzw_to_rotmat(q):
    x, y, z, w = _normalize_quaternion_xyzw(q)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _rotmat_to_quat_xyzw(R):
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))

    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return _normalize_quaternion_xyzw(np.array([x, y, z, w], dtype=np.float64))


def _parse_delta_pose(delta_pose):
    arr = np.asarray(delta_pose, dtype=np.float64)

    if arr.shape == (4, 4):
        R = arr[:3, :3]
        t = arr[:3, 3]
        return R, t

    arr = arr.reshape(-1)
    if arr.shape[0] == 7:
        t = arr[:3]
        q_xyzw = arr[3:]
        R = _quat_xyzw_to_rotmat(q_xyzw)
        return R, t

    raise ValueError("delta_pose must be shape (4,4) or length-7 [tx,ty,tz,qx,qy,qz,qw]")


def _slerp_identity_to_quat_batch(q1_xyzw, alpha):
    q1 = _normalize_quaternion_xyzw(q1_xyzw)

    # Keep shortest path to identity quaternion.
    if q1[3] < 0.0:
        q1 = -q1

    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)

    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.dot(q0, q1))
    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        q = (1.0 - alpha)[:, None] * q0[None, :] + alpha[:, None] * q1[None, :]
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return q

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    w0 = np.sin((1.0 - alpha) * theta) / sin_theta
    w1 = np.sin(alpha * theta) / sin_theta

    q = w0[:, None] * q0[None, :] + w1[:, None] * q1[None, :]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def _quat_xyzw_batch_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def undistort_points(points_with_perpoint_time, delta_pose):
    """
    Motion-compensate LiDAR points to the scan end frame.

    Args:
        points_with_perpoint_time: (N,4) array [x,y,z,t], where t is per-point time
            over the scan interval (arbitrary scale). The function normalizes t
            to [0,1] using min/max in this frame.
        delta_pose: Relative pose from scan start frame to scan end frame,
            expressed as T_end_from_start, either
            - (4,4) homogeneous matrix, or
            - length-7 [tx, ty, tz, qx, qy, qz, qw].

            For world-frame sensor poses T_world_from_sensor:
                T_end_from_start = inv(T_world_from_sensor_end) @ T_world_from_sensor_start

    Returns:
        (N,4) float32 array [x_undist, y_undist, z_undist, t].
    """
    pts_t = np.asarray(points_with_perpoint_time, dtype=np.float64)
    if pts_t.ndim != 2 or pts_t.shape[1] < 4:
        raise ValueError("points_with_perpoint_time must be an (N,4+) array [x,y,z,t]")

    xyz = pts_t[:, :3]
    t = pts_t[:, 3]

    n = xyz.shape[0]
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)

    R_delta, t_delta = _parse_delta_pose(delta_pose)

    t_min = float(np.min(t))
    t_max = float(np.max(t))
    if (t_max - t_min) < _EPS:
        alpha = np.zeros_like(t, dtype=np.float64)
    else:
        alpha = (t - t_min) / (t_max - t_min)
        alpha = np.clip(alpha, 0.0, 1.0)

    # Pose at each point time, interpolated from start (I) to end (delta).
    q_delta = _rotmat_to_quat_xyzw(R_delta)
    q_alpha = _slerp_identity_to_quat_batch(q_delta, alpha)
    R_start_i = _quat_xyzw_batch_to_rotmat(q_alpha)
    t_start_i = alpha[:, None] * t_delta[None, :]

    # Transform point from its capture frame i to end frame:
    # T_start_end maps start->end and T_start_i maps start->i.
    # Therefore: T_i_end = T_start_end @ inv(T_start_i).
    R_i_T = np.transpose(R_start_i, (0, 2, 1))
    R_i_end = np.einsum("jk,nkl->njl", R_delta, R_i_T)
    t_i_end = t_delta[None, :] - np.einsum("nij,nj->ni", R_i_end, t_start_i)

    xyz_undist = np.einsum("nij,nj->ni", R_i_end, xyz) + t_i_end

    out = np.empty((n, 4), dtype=np.float32)
    out[:, :3] = xyz_undist.astype(np.float32)
    out[:, 3] = t.astype(np.float32)
    return out

def convert_time_to_rgb(points_with_perpoint_time):
    '''
    points_with_perpoint_time: (N,4) array [x,y,z,t], where t is per-point time
            over the scan interval (arbitrary scale). The function normalizes t
            to [0,1] using min/max in this frame.
    time range(a, b) -> rgb color range(0, 255)
    output: (N,3) array [r,g,b], where r,g,b are in range [0, 255], use jet colormap to map time to color
    '''
    import matplotlib.pyplot as plt
    pts_t = np.asarray(points_with_perpoint_time, dtype=np.float64)
    if pts_t.ndim != 2 or pts_t.shape[1] < 4:
        raise ValueError("points_with_perpoint_time must be an (N,4+) array [x,y,z,t]")
    t = pts_t[:, 3]
    t_min = np.min(t)
    t_max = np.max(t)
    if (t_max - t_min) < 1e-8:
        t_normalized = np.zeros_like(t, dtype=np.float64)
    else:
        t_normalized = (t - t_min) / (t_max - t_min)

    rgb = (t_normalized * 255).astype(np.uint8)
    # map to jet colormap
    rgb = plt.cm.jet(rgb)[:, :3] * 255
    rgb = rgb.astype(np.uint8)
    return rgb

if __name__ == "__main__":
    from test_projection import pose7_to_matrix, interpolate_pose_based_on_timestamps, pointcloud_transform
    import rerun as rr

    points_folder = '/home/wenshan/tmp/offroad_test/00_test_rtk_spoofer_wfig8_med_rnn/sensors/velodyne_1/'
    timestamp_file = f'{points_folder}/timestamps.txt'
    velodyne_timestamps = np.loadtxt(timestamp_file)


    # original_points_folder = '/home/wenshan/tmp/offroad_test/07_all_marsh/sensors/velodyne_1/'
    # original_timestamp_file = f'{original_points_folder}/timestamps.txt'

    rtk_file = '/home/wenshan/tmp/offroad_test/00_test_rtk_spoofer_wfig8_med_rnn/sensors/gq7_rtk_odometry/interp_data.txt'
    rtk_timestamps_file = '/home/wenshan/tmp/offroad_test/00_test_rtk_spoofer_wfig8_med_rnn/sensors/gq7_rtk_odometry/interp_timestamps.txt'
    rtk_timestamps = np.loadtxt(rtk_timestamps_file)
    rtk_data = np.loadtxt(rtk_file)

    # original_timestamps = np.loadtxt(original_timestamp_file)

    # diff = timestamps[1:] - timestamps[:-1]
    # original_diff = original_timestamps[1:] - original_timestamps[:-1]

    # rtk_diff = rtk_timestamps[1:] - rtk_timestamps[:-1]


    rtk_poses = np.array([pose7_to_matrix(pose) for pose in rtk_data[:, :7]])
    rtk_poses_transformed = np.array([np.linalg.inv(rtk_poses[0]) @ pose for pose in rtk_poses])

    timeoffset = 0.0
    # velodyne_poses = interpolate_pose_based_on_timestamps(rtk_timestamps, rtk_poses_transformed, velodyne_timestamps+timeoffset)

    index = 242
    points = np.load(f'{points_folder}/{index:08d}.npy')

    endtime_points = velodyne_timestamps[index]
    starttime_points = velodyne_timestamps[index] - 0.1
    pose_start = interpolate_pose_based_on_timestamps(
        rtk_timestamps, rtk_poses_transformed, np.array([starttime_points + timeoffset])
    )[0]
    pose_end = interpolate_pose_based_on_timestamps(
        rtk_timestamps, rtk_poses_transformed, np.array([endtime_points + timeoffset])
    )[0]
    # Relative motion over the scan: start -> end.
    delta_pose = np.linalg.inv(pose_end) @ pose_start

    undistorted_points = undistort_points(points, delta_pose)

    # compare undistorted points with the superodometry points
    so_points_folder = '/home/wenshan/tmp/offroad_test/00_test_rtk_spoofer_wfig8_med_rnn/super_odometry/pointcloud_in_odom/'
    so_points = np.load(f'{so_points_folder}/{index+1:08d}.npy')
    so_poses = np.loadtxt('/home/wenshan/tmp/offroad_test/00_test_rtk_spoofer_wfig8_med_rnn/super_odometry/odometry/data.txt')
    so_points_transformed = pointcloud_transform(so_points[:, :3], np.linalg.inv(np.linalg.inv(pose7_to_matrix(so_poses[0][:7])) @ pose7_to_matrix(so_poses[index+1][:7])))

    # visualize points with time in rerun
    rgb = convert_time_to_rgb(points)
    rr.init('rtk_lidar_registration', spawn=True)

    # compare the results in rerun
    rr.init('rtk_lidar_registration', spawn=True)
    rr.log(f"pointcloud/pointcloud_{index}", rr.Points3D(undistorted_points[:, :3], colors=rgb, radii=0.03))
    
    rr.log(f"pointcloud/pointcloud_{index}_original", rr.Points3D(points[:, :3], colors=rgb, radii=0.01))
    # rr.log(f"pointcloud/pointcloud_{index}_so", rr.Points3D(so_points_transformed[:, :3], colors=[[0, 0, 255]], radii=0.01))

    # import matplotlib.pyplot as plt
    # plt.plot(diff, label='point cloud with time')
    # plt.plot(original_diff, label='original point cloud')
    # plt.plot(rtk_diff, label='rtk odometry')
    # plt.legend()
    # plt.show()

    import ipdb; ipdb.set_trace()