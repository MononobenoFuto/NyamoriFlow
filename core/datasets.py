# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.transforms import Resize 


import os
import math
import cv2
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor, SparseMeshAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            
            if len(img1.shape) == 2:
                img1 = np.expand_dims(img1,2).repeat(3,axis=2)
                img2 = np.expand_dims(img2,2).repeat(3,axis=2)
            else:    
                img1 = img1[..., :3]
                img2 = img2[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float(), self.extra_info[index]


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/home/user2/dataset/opticalflow/MPI-Sintel-complete', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='/home/user2/dataset/opticalflow/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/home/user2/dataset/opticalflow/flyingthings3d_complete', dstype='frames_cleanpass', split='training'):
        super(FlyingThings3D, self).__init__(aug_params)

        split_dir = 'TRAIN' if split == 'training' else 'TEST'
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, f'{split_dir}/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, f'optical_flow/{split_dir}/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]

class FlyingThings3DDistract(FlowDataset):
    def __init__(self, aug_params=None, root='/home/user2/dataset/opticalflow/flyingthings3d_complete', dstype='frames_cleanpass', split='training'):
        super(FlyingThings3DDistract, self).__init__(aug_params)

        split_dir = 'TRAIN' if split == 'training' else 'TEST'
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, f'{split_dir}/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, f'optical_flow/{split_dir}/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(1, len(flows)-2):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1], images[i-1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i], images[i+2]] ]
                            self.flow_list += [ flows[i+1] ]
    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img_distract = frame_utils.read_gen(self.image_list[index][2])
            
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]

            img_distract = np.array(img_distract).astype(np.uint8)[..., :3]

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            img_distract = torch.from_numpy(img_distract).permute(2, 0, 1).float()
            img2 = img_distract * 0.3 + img2 * 0.7
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img_distract = frame_utils.read_gen(self.image_list[index][2])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.float32)
        img_distract = np.array(img_distract).astype(np.float32)

        dis_rate = random.uniform(0, 0.5)
        img2 = img2 * (1 - dis_rate) + img_distract * dis_rate

        img2 = img2.astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()
      


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/home/why/vslam/dataset/kitti', dirName="image_2"):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, dirName, '*_10.png')))
        images2 = sorted(glob(osp.join(root, dirName, '*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        
class KITTI_RAW(FlowDataset):
    def __init__(self, aug_params=None, root='/home/why/vslam/dataset/kitti_raw/2011_09_30_drive_0018_sync/image_00/data'):
        super(KITTI_RAW, self).__init__(aug_params, sparse=True)
        self.is_test = True

        images = sorted(glob(osp.join(root, '*.png')))

        for i in range(len(images)-1):
            img1 = images[i]
            img2 = images[i+1]
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]
        print(len(images))

class KITTI_RAW_GT(FlowDataset):
    def __init__(self, aug_params=None, root=None):
        super(KITTI_RAW_GT, self).__init__(aug_params, sparse=True)

        self.is_test = False
        self.pose_list = []

        if self.is_test:
            rt = f'/home/why/vslam/dataset/kitti_raw/2011_09_30_drive_0034_sync/image_00/data'
            images = sorted(glob(osp.join(rt, '*.png')))
            for i in range(len(images)-1):
                img1 = images[i]
                img2 = images[i+1]
                frame_id = img1.split('/')[-1]
                self.extra_info += [ [frame_id] ]
                self.image_list += [ [img1, img2] ]
            print(f'test dataset len: {len(self.image_list)}')

        else:
            n_list = ['0016', '0018', '0020', '0027', '0028', '0033', '0034']
            p_list = ['04', '05', '06', '07', '08', '09', '10']
            for j in range(len(n_list)):
                rt = f'/home/why/vslam/dataset/kitti_raw/2011_09_30_drive_{n_list[j]}_sync/image_00/data'
                
                images = sorted(glob(osp.join(rt, '*.png')))
                pose_path = f'/home/why/vslam/dataset/kitti_raw/gt_camera/{p_list[j]}.txt'

                f = open(pose_path, 'r')
                poses = f.readlines()
                for i in range(0, len(poses)-1, 5):
                    img1 = images[i]
                    img2 = images[i+1]
                    frame_id = img1.split('/')[-1]
                    self.extra_info += [ [frame_id] ]
                    self.image_list += [ [img1, img2]]
                    self.pose_list  += [ [poses[i], poses[i+1]]]
                # print(f'reading info {p_list[j]}: {(len(poses)-1) / len(images)}')     

            self.augmentor = SparseMeshAugmentor(**aug_params)

            grid_x, grid_y = np.meshgrid(np.arange(aug_params['crop_size'][1]), np.arange(aug_params['crop_size'][0]))
            self.meshgrid = np.float64(np.stack((grid_x, grid_y), axis=-1))
            print(f'len of pose_list:{len(self.pose_list)}')

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
            
            if len(img1.shape) == 2:
                img1 = np.expand_dims(img1,2).repeat(3,axis=2)
                img2 = np.expand_dims(img2,2).repeat(3,axis=2)
            else:    
                img1 = img1[..., :3]
                img2 = img2[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % self.__len__()
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        
        #>t1增强前>
        img_before = np.repeat(img1[:,:,np.newaxis], 3, axis=2)
        print(f'img_before shape:{img_before.shape}')

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            img1, img2, correction = self.augmentor(img1, img2)
            #展示数据增强造成的仿射变换是否准确

            print(f'cor = {correction}')

        #>t1增强后
        img_after = np.pad(img1, ((0,0), (0, 1226-720), (0,0)), 'constant', constant_values=0)
        img_show = np.concatenate((img_before, img_after), axis=0)

        n_af_p = 30
        x2 = np.random.randint(0, 720, size=n_af_p)
        y2 = np.random.randint(0, 370, size=n_af_p)
        x1 = (x2 - correction[2]) / correction[0]
        y1 = (y2 - correction[3]) / correction[1]
        print(f'x1shape:{x1.shape}, x2shape:{x2.shape}')

        for i in range(n_af_p):
            _x, _y = int(x1[i]), int(y1[i])
            if _x >= 0 and _x < 1226 and _y >= 0 and _y <= 370:
                cv2.line(img_show, (_x,_y), (x2[i],y2[i]+370), color=(0,0,255))

        cv2.imwrite('/home/why/vslam/SAMFLow-test/pretrain/show_affine/'+str(index)+'.png', img_show)
        print(f'img_after shape:{img_after.shape}')
        #>t1 结束

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()


        pose1 = torch.Tensor([float(x) for x in self.pose_list[index][0].split()])
        pose2 = torch.Tensor([float(x) for x in self.pose_list[index][1].split()])

        return img1, img2, self.meshgrid, torch.tensor(correction), pose1, pose2

    def __len__(self):
        return len(self.image_list)


class TUM(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/home/why/vslam/dataset/tum/rgbd_dataset_freiburg2_desk_with_person'):
        super(TUM, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        
        images = sorted(glob(osp.join(root, 'rgb', '*.png')))

        for i in range(len(images)-1):
            img1 = images[i]
            img2 = images[i+1]
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        print('image number', len(self.image_list))
        print(self.image_list[0])
        if split == "training":
            pass
        # root = osp.join(root, split)
        # images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        # images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        # for img1, img2 in zip(images1, images2):
        #     frame_id = img1.split('/')[-1]
        #     self.extra_info += [ [frame_id] ]
        #     self.image_list += [ [img1, img2] ]

        # if split == 'training':
        #     self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='/home/user2/dataset/opticalflow/HD1K'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things' or args.stage == 'things_kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset
    
    elif args.stage == 'things_distract':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        distrac_dataset = FlyingThings3DDistract(aug_params, dstype='frames_cleanpass')
        train_dataset = clean_dataset + final_dataset + distrac_dataset

    elif args.stage == 'sintel' or args.stage == 'sintel_kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        things_f = FlyingThings3D(aug_params, dstype='frames_finalpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 200*sintel_clean + 200*sintel_final + 400*kitti + 10*hd1k + things + things_f

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    elif args.stage == 'kitti_raw_polarloss':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI_RAW_GT(aug_params)


    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=8, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader
