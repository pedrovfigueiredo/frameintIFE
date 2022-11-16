import torch
import torch.nn as nn
from loss.vggloss import VGGLoss

class BlendingLoss(nn.Module):
    def __init__(self, loss_type=nn.MSELoss(), loss_weights={'warping' : 1.0}):
        super().__init__()
        self.loss_weights = loss_weights
        self.loss = loss_type

        if 'vgg' in loss_weights:
            self.vgg = VGGLoss()

    def forward(self, net_out, gt_buffers):
        loss_dict = {}

        # compute the relit image loss.
        if 'reconst' in self.loss_weights:
            interp_gt = gt_buffers['reference_img']
            interp_img = net_out['out_img']
            loss_dict['reconst'] = self.loss(interp_img, interp_gt)

        if 'vgg' in self.loss_weights:
            interp_gt = gt_buffers['reference_img']
            interp_img = net_out['out_img']
            loss_dict['vgg'] = self.vgg(interp_img, interp_gt)

        # compute the warping loss.
        if 'warping' in self.loss_weights:
            interp_gt = gt_buffers['reference_img']
            interp_l = net_out['out_l']
            interp_r = net_out['out_r']
            loss_dict['warping'] = (self.loss(interp_l, interp_gt) + self.loss(interp_r, interp_gt)) / 2.0

        return loss_dict

    def get_loss_weights(self):
        return self.loss_weights