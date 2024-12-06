import numpy as np
from math import pi
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.fft as fft
# import matplotlib.pyplot as plt
# import matplotlib.transforms as mtransforms
# import scipy
# import scipy.io as sio
# from scipy import ndimage
# import tifffile
from misc import show_imset, circular_aper, ZernikeBasis, mask_resize_reshape
# import PIL

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

class LensModel(nn.Module):
    def __init__(self, model_dict):
        super(LensModel, self).__init__()
        self.N = model_dict['N']  # initial simulatio size
        self.M = model_dict['M']  # magnification
        self.ps_aper = model_dict['ps_aper']  # pixel size at aperture stop
        self.N_aper = model_dict['N_aper']  # the number of pixels along the aperture diameter
        self.ps_psf = model_dict['ps_psf']  # pixel size at image plane
        self.device = model_dict['device']  # device
        self.w05 = model_dict['w05']  # half width (in pixel) of PSF
        self.mask_phase = model_dict['mask_phase']  # mask phase at aperture stop
        self.aper = model_dict['aper']  # aperture
        self.znms = model_dict['znms']  # zernike modes/pyramid
        self.idx_defocus = model_dict['idx_defocus']  # the index of defocus in the modes
        self.zernike_defocus_only = False  # only consider the defocus zernike basis
        self.photons = model_dict['photons']  # photon count
        self.idx05 = int(self.N / 2)  # the index of the central pixel in image plane

        self.tensor0 = torch.zeros((1, 1), device=self.device)
        self.tensor1 = torch.ones((1, 1), device=self.device)
        self.tensor_eye = torch.eye(2, device=self.device)
        self.tensor001 = torch.tensor([[0, 0, 1]], device=self.device)

        self.scale_factor = torch.tensor(1.0, device=self.device, requires_grad=True)
        self.rotation_theta = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.sigma = torch.tensor(0.5, device=self.device, requires_grad=True)
        self.z2zc = nn.Sequential(nn.Linear(1, 6), nn.Sigmoid(), nn.Linear(6, 6), nn.Sigmoid(),
                                  nn.Linear(6, self.znms.shape[0]))

        # blur parameters
        self.g_size = 7
        self.g_r = int(self.g_size / 2)
        g_xs = torch.linspace(-self.g_r, self.g_r, self.g_size, device=self.device)
        self.g_xx, self.g_yy = torch.meshgrid([g_xs, g_xs], indexing='xy')

    def rotation(self, ccw_theta):  # z-agnostic, counter-clockwise theta
        rotation_matrix = torch.cat(
            (torch.cat((torch.cos(ccw_theta.view(1, 1)), -torch.sin(ccw_theta.view(1, 1)), self.tensor0), dim=1),
             torch.cat((torch.sin(ccw_theta.view(1, 1)), torch.cos(ccw_theta.view(1, 1)), self.tensor0), dim=1),
             self.tensor001),
            dim=0)
        return rotation_matrix

    def scaling(self, scale_factor):  # z-agnostic, <1--zoom in
        scaling_matrix = torch.cat((torch.cat((scale_factor.view(1, 1), self.tensor0, self.tensor0), dim=1),
                                    torch.cat((self.tensor0, scale_factor.view(1, 1), self.tensor0), dim=1),
                                    self.tensor001),
                                   dim=0)
        return scaling_matrix

    def blur_kernel(self):
        return torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / self.sigma ** 2)

    def update(self):
        # 1 replace the scaling affine transform by zero padding or cropping
        # 2 x and y grid for lateral shifting

        N = self.N
        N2 = np.floor(N / self.scale_factor.detach().item())  # new simulation size
        N2 = int(N2 + (1 - N2 % 2))  # make it odd
        self.N2 = N2
        print(f"new simulation size: {N2}")

        aper2 = self.aper.cpu().numpy()
        znms2 = self.znms.cpu().numpy()
        if N2 >= N:  # zero padding
            before = int((N2 - N) / 2)
            after = N2 - N - before
            aper2 = np.pad(aper2, (before, after))
            znms2 = np.pad(znms2, ((0, 0), (before, after), (before, after)))
        else:  # cropping
            crop_start = int((N - N2) / 2)
            aper2 = aper2[crop_start:crop_start + N2, crop_start:crop_start + N2]
            znms2 = znms2[:, crop_start:crop_start + N2, crop_start:crop_start + N2]

        self.aper2 = torch.tensor(aper2).to(self.device)
        self.znms2 = torch.tensor(znms2).to(self.device)
        self.idx05_2 = int(N2 / 2)
        self.mask_phase2 = torch.tensor(aper2, requires_grad=True, device=self.device)  # another phase mask

        xl = torch.linspace(-1, 1, N2, device=self.device) * self.ps_aper * N2 / 2
        xx2, yy2 = torch.meshgrid(xl, xl, indexing='xy')
        self.xx2, self.yy2 = xx2, yy2  # gradient: ps_mask
        self.xgrid = self.xx2 / self.ps_aper * 2 * pi / self.N2 / self.ps_psf * self.M
        self.ygrid = self.yy2 / self.ps_aper * 2 * pi / self.N2 / self.ps_psf * self.M

    def apply(self, xyzs, new_mask_phase=None, zs_scaling=1):  # after update
        # xyzs, rank 2, [n, 3]
        # apply the mask if given
        # zs_scaling, for lens with multiple focusing distances, can be ignored

        xy, z = (xyzs[:, :2]), (xyzs[:, 2]).unsqueeze(1)
        xy_phase = (xy[:, 0]).unsqueeze(1).unsqueeze(1) * self.xgrid + (xy[:, 1]).unsqueeze(1).unsqueeze(1) * self.ygrid

        zc = self.z2zc(z)  # Bx1--> Bxzn
        zc[:, self.idx_defocus] = zc[:, self.idx_defocus] / zs_scaling
        self.current_zc = zc.detach()  # recording
        if self.zernike_defocus_only:
            z_phase = torch.sum(self.znms2[self.idx_defocus:self.idx_defocus + 1, :, :] * (
                zc[:, self.idx_defocus:self.idx_defocus + 1]).unsqueeze(2).unsqueeze(2), dim=1)
        else:
            z_phase = torch.sum(self.znms2 * zc.unsqueeze(2).unsqueeze(2), dim=1)

        if new_mask_phase is None:
            uin = self.aper2 * torch.exp(1j * (z_phase + xy_phase + self.mask_phase2))
        else:
            uin = self.aper2 * torch.exp(1j * (z_phase + xy_phase + new_mask_phase))

        uout = fft.fftshift(fft.fftn(fft.ifftshift(uin, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))
        psf = torch.abs(uout) ** 2
        psf = psf.unsqueeze(1).type(torch.float32)
        # crop
        psf = psf[:, :, self.idx05_2 - self.w05:self.idx05_2 + self.w05 + 1,
              self.idx05_2 - self.w05:self.idx05_2 + self.w05 + 1]
        # blur
        psf = F.conv2d(psf, self.blur_kernel().unsqueeze(0).unsqueeze(0), padding=self.g_r)
        # photon normalization
        psf = psf / torch.sum(psf, dim=(2, 3), keepdim=True) * self.photons  # rank 4

        return psf

    def forward(self, xyzs):  # xyzs-Bx3
        # xyzs, rank 2, [n, 3]
        z = (xyzs[:, 2]).unsqueeze(1)
        # z phase
        zc = self.z2zc(z)  # Bx1--> Bxzn
        self.current_zc = zc.detach()
        if self.zernike_defocus_only:
            z_phase = torch.sum(self.znms[self.idx_defocus:self.idx_defocus + 1, :, :] * (
                zc[:, self.idx_defocus:self.idx_defocus + 1]).unsqueeze(2).unsqueeze(2), dim=1)
        else:
            z_phase = torch.sum(self.znms * zc.unsqueeze(2).unsqueeze(2), dim=1)

        # input electric field
        uin = self.aper * torch.exp(1j * (z_phase + self.mask_phase))
        uout = fft.fftshift(fft.fftn(fft.ifftshift(uin, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))
        psf = torch.abs(uout) ** 2
        psf = psf.unsqueeze(1).type(torch.float32)  # for affine transform

        # crop
        psf = psf[:, :, self.idx05 - self.w05:self.idx05 + self.w05 + 1,
              self.idx05 - self.w05:self.idx05 + self.w05 + 1]

        # affine transform
        sr_m = self.scaling(self.scale_factor) @ self.rotation(self.rotation_theta)
        sr_m = sr_m.unsqueeze(0).repeat(psf.shape[0], 1, 1)
        grid = F.affine_grid(sr_m[:, :2, :], psf.size(), align_corners=True)
        psf = F.grid_sample(psf, grid, align_corners=True)

        # blur
        psf = F.conv2d(psf, self.blur_kernel().unsqueeze(0).unsqueeze(0), padding=self.g_r)

        # photon normalization
        psf = psf / torch.sum(psf, dim=(2, 3), keepdim=True) * self.photons

        return psf


class Dataset(torch.utils.data.Dataset):
    def __init__(self, xyzs, psfs):
        self.xyzs = xyzs
        self.psfs = psfs

    def __len__(self):
        return self.xyzs.shape[0]

    def __getitem__(self, idx):
        x = self.xyzs[idx, :]
        y = self.psfs[idx, :, :]
        return x, y


def calculate_cc(output, target):
    # output: rank 3, target: rank 3
    output_mean = np.mean(output, axis=(1, 2), keepdims=True)
    target_mean = np.mean(target, axis=(1, 2), keepdims=True)
    ccs = (np.sum((output - output_mean) * (target - target_mean), axis=(1, 2)) /
           (np.sqrt(np.sum((output - output_mean) ** 2, axis=(1, 2)) * np.sum((target - target_mean) ** 2,
                                                                              axis=(1, 2))) + 1e-9))
    return ccs


def calculate_cc_v2(output, target, r=1):
    # compensate for the centering issue
    # output: ndarray, rank 3
    # target: ndarray, rank 3
    # r: compensation radius
    m_size = target.shape[1]
    m_size_new = m_size - r * 2
    ccs = []
    for i in range(output.shape[0]):
        m0 = target[i, :, :]
        m1 = output[i, :, :]
        sub_cc = []
        for ii in np.arange(-r, r + 1):
            for jj in np.arange(-r, r + 1):
                cc = calculate_cc(output[i:i + 1, r + ii:r + ii + m_size_new, r + jj:r + jj + m_size_new],
                                  target[i:i + 1, r:m_size - r, r:m_size - r])
                sub_cc.append(cc[0])
        ccs.append(max(sub_cc))
    return np.array(ccs)


def xyzs_for_crlb(xyzs, dx, dy, dz):
    nz = xyzs.shape[0]
    crlb_xyzs = xyzs.repeat(4, 1)
    crlb_xyzs[1 * nz:2 * nz, 0] = crlb_xyzs[1 * nz:2 * nz, 0] + dx
    crlb_xyzs[2 * nz:3 * nz, 1] = crlb_xyzs[2 * nz:3 * nz, 1] + dy
    crlb_xyzs[3 * nz:4 * nz, 2] = crlb_xyzs[3 * nz:4 * nz, 2] + dz
    return crlb_xyzs


def crlb_calculator(psfs, nz, dx, dy, dz, beta):

    psfs0 = psfs[:1 * nz, 0, :, :]
    dpdx = (psfs[1 * nz:2 * nz, 0, :, :] - psfs0) / dx
    dpdy = (psfs[2 * nz:3 * nz, 0, :, :] - psfs0) / dy
    dpdz = (psfs[3 * nz:4 * nz, 0, :, :] - psfs0) / dz
    dxx = torch.sum(dpdx * dpdx / (psfs0 + beta + 1e-9), dim=(1, 2), keepdim=True)
    dxy = torch.sum(dpdx * dpdy / (psfs0 + beta + 1e-9), dim=(1, 2), keepdim=True)
    dxz = torch.sum(dpdx * dpdz / (psfs0 + beta + 1e-9), dim=(1, 2), keepdim=True)

    dyy = torch.sum(dpdy * dpdy / (psfs0 + beta + 1e-9), dim=(1, 2), keepdim=True)
    dyz = torch.sum(dpdy * dpdz / (psfs0 + beta + 1e-9), dim=(1, 2), keepdim=True)

    dzz = torch.sum(dpdz * dpdz / (psfs0 + beta + 1e-9), dim=(1, 2), keepdim=True)

    finfo = torch.cat((torch.cat((dxx, dxy, dxz), dim=2),
                       torch.cat((dxy, dyy, dyz), dim=2),
                       torch.cat((dxz, dyz, dzz), dim=2)),
                      dim=1)

    crlb = torch.sqrt(torch.diagonal(torch.linalg.inv(finfo), dim1=1, dim2=2)) * 1e6  # um
    return crlb


def TV(M):
    return (M[1:, :] - M[:-1, :]).pow(2).sum() + (M[:, 1:] - M[:, :-1]).pow(2).sum()