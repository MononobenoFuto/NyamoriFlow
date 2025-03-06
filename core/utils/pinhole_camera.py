import numpy as np
import torch
from scipy.spatial.transform import Rotation

class PinholeCamera():
    def __init__(self, k1, k2, p1, p2, fx, fy, cx, cy):
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        if k1 == 0.0 and k2 == 0.0 and p1 == 0.0 and p2 == 0.0:
            self.m_noDistortion = True
        else:
            self.m_noDistortion = False

        self.m_inv_K11 = 1.0 / fx
        self.m_inv_K13 = -cx / fx
        self.m_inv_K22 = 1.0 / fy
        self.m_inv_K23 = -cy / fy
        
    def liftProjective(self, p2):
        mx_d = self.m_inv_K11 * p2[0] + self.m_inv_K13
        my_d = self.m_inv_K22 * p2[1] + self.m_inv_K23

        if self.m_noDistortion:
            mx_u = mx_d
            my_u = my_d
        else:
            n = 8
            d_u = self.distortion([mx_d, my_d])
            mx_u = mx_d - d_u[0]
            my_u = my_d - d_u[1]

            for i in range(n-1):
                d_u = self.distortion([mx_u, my_u])
                mx_u -= d_u[0]
                my_u -= d_u[1]
        return [mx_u, my_u, 1.0]

    def distortion(self, p_u):
        mx2_u = p_u[0] * p_u[0]
        my2_u = p_u[1] * p_u[1]
        mxy_u = p_u[0] * p_u[1]
        rho2_u = mx2_u + my2_u
        rad_dist_u = self.k1 * rho2_u + self.k2 * rho2_u * rho2_u

        ret_x = p_u[0] * rad_dist_u + 2.0 * self.p1 * mxy_u + self.p2 * (rho2_u + 2.0 * mx2_u)
        ret_y = p_u[1] * rad_dist_u + 2.0 * self.p2 * mxy_u + self.p1 * (rho2_u + 2.0 * my2_u)
        return [ret_x, ret_y]

def lift(p):
    fx = 7.070912e+02
    fy = 7.070912e+02
    cx = 6.018873e+02
    cy = 1.831104e+02
    p[:,0,:,:] = 1.0 / fx * p[:,0,:,:] - cx / fx
    p[:,1,:,:] = 1.0 / fy * p[:,1,:,:] - cy / fy


    nc = torch.ones(p.shape[0], 1, p.shape[2], p.shape[3])
    nc = nc.cuda()

    return torch.cat((p, nc), dim=1)


def get_epipolar_line(p1, p2, pose1, pose2):
    R1 = torch.from_numpy(Rotation.from_quat(pose1[0][4:].cpu().numpy()).as_matrix())
    R2 = torch.from_numpy(Rotation.from_quat(pose2[0][4:].cpu().numpy()).as_matrix())
    R1 = R1.cuda()
    R2 = R2.cuda()

    dt = (pose2[:,1:4] - pose1[:,1:4]).cuda()
    l1 = torch.einsum('ij,bjkl->bikl', R1, p1)
    l2 = torch.einsum('ij,bjkl->bikl', R2, p2)
    cvl = torch.cross(l1, l2, dim=1)
    # l1 = torch.matmul(p1.view(-1, 3), R1.T)
    # l2 = torch.matmul(p2.view(-1, 3), R2.T)
    # cvl = torch.cross(l1, l2)
    
    dt = dt.unsqueeze(2).unsqueeze(3)
    ue = torch.sum(dt * cvl, dim=0).abs()
    shita = torch.norm(cvl, dim=1)

    # ue = torch.sum(dt * cvl, dim=-1).abs()
    # shita = torch.norm(cvl, dim=-1)
    # print(f'{ue.shape} / {shita.shape}')
    return ue / shita