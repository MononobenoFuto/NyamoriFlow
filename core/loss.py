import torch
from utils.pinhole_camera import lift, get_epipolar_line
import time

MAX_FLOW = 400

def sequence_loss(flow_preds, flow_gt, valid, cfg):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += torch.nan_to_num(i_weight * (valid[:, None] * i_loss).mean(), 0.0, 0.0, 0.0)
    
    flow_loss = torch.nan_to_num(flow_loss, 0.0, 0.0, 0.0)

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    flow_gt_length = torch.sum(flow_gt**2, dim=1).sqrt()
    flow_gt_length = flow_gt_length.view(-1)[valid.view(-1)]
    for t in flow_gt_thresholds:
        e = epe[flow_gt_length < t]
        metrics.update({
                f"{t}-th-5px": (e < 5).float().mean().item()
        })


    return flow_loss, metrics

def polar_loss(flow_preds, mesh, cr, pose1, pose2, cfg):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    time1 = time.time()

    #变换前坐标
    mesh = mesh.permute(0, 3, 1, 2)

    position_before = torch.empty_like(mesh)
    position_before[:,0,:,:] = (mesh[:,0,:,:] - cr[0,2]) / cr[0,0]
    position_before[:,1,:,:] = (mesh[:,1,:,:] - cr[0,3]) / cr[0,1]
    position_before = lift(position_before)


    # print(f'postion_before_shape:{position_before.shape}')
    # print(f'mesh_shape:{mesh.shape}')
    # print(f'flow_shape:{flow_preds[0].shape}')
    # print(f'pb: {position_before.is_cuda}')


    position_after = torch.empty_like(mesh)
    for i in range(n_predictions):
        position_after = (mesh + flow_preds[i])
        # print(f'pr dtype:{position_after.dtype}')
        # print(f'pa: {position_after.is_cuda}')
        # print(f'postion_after_shape:{ position_after.shape}')
        position_after[:,0,:,:] = (position_after[:,0,:,:] - cr[0,2]) / cr[0,0]
        position_after[:,1,:,:] = (position_after[:,1,:,:] - cr[0,3]) / cr[0,1]
        position_after = lift(position_after)

        

        i_weight = gamma**(n_predictions - i - 1)
        i_loss = get_epipolar_line(position_before, position_after, pose1, pose2)
        # print(i_loss.mean())
        flow_loss += torch.nan_to_num(i_weight * i_loss.mean(), 0.0, 0.0, 0.0)
    
    flow_loss = torch.nan_to_num(flow_loss, 0.0, 0.0, 0.0)
    time2 = time.time()
    print(f'loss compute time:{time2-time1}')
    return flow_loss