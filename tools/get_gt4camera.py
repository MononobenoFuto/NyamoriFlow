import os
import glob
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np


root_path = '/home/why/vslam/dataset/kitti_raw/'

seq_name_list = ['', '', '', '', '2011_09_30_drive_0016_sync',
    '2011_09_30_drive_0018_sync',
    '2011_09_30_drive_0020_sync',
    '2011_09_30_drive_0027_sync',
    '2011_09_30_drive_0028_sync',
    '2011_09_30_drive_0033_sync',
    '2011_09_30_drive_0034_sync']

seq_id_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']


for i in range(4, 11, 1):
    gt_times = []
    gt_p = []
    gt_q = []
    with open(f'/home/why/vslam/dataset/kitti_raw/gt_tum/{seq_id_list[i]}_gt_tum.txt') as f:
        info = f.readlines()
        cc = 0
        for line in info:
            cc += 1
            a, b, c, d, e, f, g, h = line.split()
            gt_times.append(float(a))
            gt_p.append([float(b), float(c), float(d)])
            gt_q.append([float(e), float(f), float(g), float(h)])
    gt_p = np.array(gt_p)
    gt_q = np.array(gt_q)
    gt_q_rot = R.from_quat(gt_q)

    inter_x = interp1d(gt_times, gt_p[:, 0])
    inter_y = interp1d(gt_times, gt_p[:, 1])
    inter_z = interp1d(gt_times, gt_p[:, 2])
    slerp = Slerp(gt_times, gt_q_rot)


    camera_times = []
    with open(f'/home/why/vslam/dataset/kitti_raw/{seq_name_list[i]}/image_00/timestamps.txt', 'r') as f:
        camera_times_raw = f.readlines()
        zero = 0.0
        for time in camera_times_raw:
            x = time.split()[-1].rstrip()
            h, m, s = x.split(':')
            h, m, s = float(h), float(m), float(s)

            time = h * 3600.0 + m * 60.0 + s

            if len(camera_times) == 0:
                zero = time
            camera_times.append(time-zero)

    end_flag = len(camera_times)
    while camera_times[end_flag-1] > gt_times[-1] - 0.00001:
        end_flag -= 1

    new_x = inter_x(camera_times[:end_flag])
    new_y = inter_y(camera_times[:end_flag])
    new_z = inter_z(camera_times[:end_flag])
    new_q = slerp(camera_times[:end_flag])
    new_q_quat = new_q.as_quat()
    with open(f'/home/why/vslam/dataset/kitti_raw/gt_camera/{seq_id_list[i]}.txt', 'w') as f:
        for j in range(end_flag):
            str = f'{camera_times[j]} {new_x[j]} {new_y[j]} {new_z[j]} {new_q_quat[j][0]} {new_q_quat[j][1]} {new_q_quat[j][2]} {new_q_quat[j][3]}\n'
            f.write(str)
    

    



