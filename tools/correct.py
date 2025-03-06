import numpy as np
import torch
from scipy.spatial.transform import Rotation
import os

def load_poses(pose_file):
    """加载相机位姿"""
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            poses.append(values)
    return np.array(poses)

def get_fundamental_matrix(pose1, pose2):
    """计算基础矩阵F"""
    t1 = pose1[1:4]
    q1 = pose1[4:]
    R1 = Rotation.from_quat([q1[0], q1[1], q1[2], q1[3]]).as_matrix()
    
    t2 = pose2[1:4]
    q2 = pose2[4:]
    R2 = Rotation.from_quat([q2[0], q2[1], q2[2], q2[3]]).as_matrix()
    
    R = R2 @ R1.T
    t = t2 - R @ t1
    
    t_cross = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = t_cross @ R
    
    K = np.array([
        [707.0912, 0, 601.8873],
        [0, 707.0912, 183.1104],
        [0, 0, 1]
    ])
    
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    return torch.from_numpy(F).cuda().float()

def compute_epipolar_correction(flow, F):
    """使用CUDA计算对极线修正"""
    device = torch.device('cuda')
    
    # 转换flow为torch tensor
    flow = torch.from_numpy(flow).cuda().float()
    h, w = flow.shape[1:]
    
    # 生成坐标网格
    y, x = torch.meshgrid(torch.arange(h, device=device), 
                         torch.arange(w, device=device),
                         indexing='ij')
    
    # 原始点坐标
    pts1 = torch.stack([x, y, torch.ones_like(x)], dim=2)  # (h, w, 3)
    
    # 光流预测点坐标
    pts2 = torch.stack([x + flow[0], y + flow[1], torch.ones_like(x)], dim=2)  # (h, w, 3)
    
    # 计算对极线 (l = F * x1)
    pts1 = pts1.float()
    lines = torch.einsum('ij,hwj->hwi', F, pts1)  # (h, w, 3)
    
    # 计算垂足
    a = lines[..., 0]
    b = lines[..., 1]
    c = lines[..., 2]
    
    x2 = pts2[..., 0]
    y2 = pts2[..., 1]
    
    denominator = a*a + b*b
    d = -(a*x2 + b*y2 + c) / denominator
    
    foot_x = x2 + a*d
    foot_y = y2 + b*d
    
    # 计算新的光流
    new_flow = torch.stack([
        foot_x - x,
        foot_y - y
    ], dim=0)
    
    return new_flow.cpu().numpy()

def process_flow(flow_dir, pose_file, batch_size=10):
    """批量处理光流文件"""
    poses = load_poses(pose_file)
    
    # 获取所有需要处理的文件
    flow_files = []
    for i in range(len(poses)-1):
        flow_path = os.path.join(flow_dir, f"{str(i).zfill(10)}.npy")
        if os.path.exists(flow_path):
            flow_files.append((i, flow_path))
    
    # 批量处理
    for i in range(0, len(flow_files), batch_size):
        batch_files = flow_files[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(flow_files)-1)//batch_size + 1}")
        
        for idx, flow_path in batch_files:
            # 加载光流
            flow = np.load(flow_path)
            
            # 计算基础矩阵
            F = get_fundamental_matrix(poses[idx], poses[idx+1])
            
            # 修正光流
            corrected_flow = compute_epipolar_correction(flow, F)
            
            # 保存修正后的光流
            save_path = os.path.join("/home/why/vslam/SAMFLow-test/pretrain/05c", f"{str(idx).zfill(10)}.npy")
            np.save(save_path, corrected_flow)
            
            print(f"Saved corrected flow to {save_path}")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # 设置设备
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch
    
    flow_dir = "/home/why/vslam/SAMFLow-test/SAMFLow-main/samflow_kittif/05"
    pose_file = "/home/why/vslam/dataset/kitti_raw/gt_camera/05.txt"
    
    # 设置批处理大小
    batch_size = 10
    process_flow(flow_dir, pose_file, batch_size)