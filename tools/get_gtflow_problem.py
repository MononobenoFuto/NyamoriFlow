import os
import glob
import numpy as np
import cv2

flow_path = '/home/why/vslam/SAMFLow-test/pretrain/20241103_flowgt/04/'
pic_path = '/home/why/vslam/dataset/kitti_raw/2011_09_30_drive_0016_sync/image_00/data/'

flow_list = glob.glob(os.path.join(flow_path, '*.npy'))

for npyn in flow_list:
    n = npyn.split('/')[-1].split('.')[0]
    x = np.load(npyn)

    picn = os.path.join(pic_path, n+ '.png')
    pic = cv2.imread(picn)
    print(x.shape)
    print(pic.shape)
    
    H, W, _ = pic.shape

    num = 100
    
    px = np.random.randint(0, W, size=num)
    py = np.random.randint(0, H, size=num)
    for i in range(num):
        tx = int(px[i] + x[0][py[i]][px[i]])
        ty = int(py[i] + x[1][py[i]][px[i]])
        # print(px[i], py[i])
        # print(x[0][py[i]][px[i]], x[1][py[i]][px[i]])
        # print(tx, ty)
        if tx > 0.0 and tx < W and ty > 0.0 and ty < H:
            cv2.line(pic, (px[i],py[i]), (tx,ty), (0,0,255), 1)

    cv2.imwrite('/home/why/vslam/SAMFLow-test/pretrain/20241105/'+n+'.png', pic)