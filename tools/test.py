import os
import glob
import cv2
import numpy as np
import math

pic_dir = '/home/why/vslam/dataset/kitti_raw/2011_09_30_drive_0034_sync/image_00/data/'
f_dir = '/home/why/vslam/SAMFLow-test/pretrain/10/'
# f_dir = '/home/why/vslam/SAMFLow-test/pretrain/20241031_flowgt/05/'
# f_dir = 'SAMFLow-test/SAMFLow-main/pretrain/04/'


def sqr(x):
    return x*x

pic_list = sorted(glob.glob(os.path.join(pic_dir, '*.png')))

mse_total = 0.0
step = 30
for i in range(0,len(pic_list)-1,step):
    p1_path = pic_list[i]
    p2_path = pic_list[i+1]

    moto = p1_path.split('/')[-1].split('.')[0]
    f_path = os.path.join(f_dir, moto+'.npy')

    img1 = cv2.imread(p1_path).astype('float32') / 255.0
    img2 = cv2.imread(p2_path).astype('float32') / 255.0

    flow = np.load(f_path)

    _, H, W = flow.shape
    # print(flow)

    mse_t = 0.0
    mse_n = 0
    for y in range(H):
        for x in range(W):
            tx = int(x + flow[0][y][x])
            ty = int(y + flow[1][y][x])
            if tx >= 0 and tx < W and ty >= 0 and ty < H:
                mse_n += 1
                mse_t += math.sqrt(sqr(img1[y][x][0]+img2[ty][tx][0]) + sqr(img1[y][x][1]+img2[ty][tx][1]) + (sqr(img1[y][x][2]+img2[ty][tx][2])))
                
    mse_t /= mse_n + 0.0
    mse_total += mse_t
mse_total /= len(pic_list)-1.0
print(mse_total * step)
