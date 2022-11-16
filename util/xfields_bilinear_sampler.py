from __future__ import absolute_import, division, print_function
import torch
from torch.nn.functional import pad

def interpolate(im, x, y, wrap_mode, tensor_type, device):
    num_batch, num_channels, height, width = im.size()
    # Handle both texture border types
    edge_size = 0
    if wrap_mode == 'border':
        edge_size = 1
        # Pad last and second-to-last dimensions by 1 from both sides
        im = pad(im, (1, 1, 1, 1))
        x = x + edge_size
        y = y + edge_size
    elif wrap_mode == 'edge':
        edge_size = 0
    else:
        return None

    # Make sure we don't go outside of image
    x = torch.clamp(x, 1.0, width - 2 + 2 * edge_size)
    y = torch.clamp(y, 1.0, height - 2 + 2 * edge_size)

    # Round disparity to sample from integer-valued pixel grid
    x0_f = torch.floor(x)
    y0_f = torch.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1

    x0 = x0_f.to(torch.float32)
    y0 = y0_f.to(torch.float32)
    
    # After rounding up we might go outside the image boundaries again
    x1 = torch.min(x1_f, torch.tensor((width - 1 + 2 * edge_size)).to(torch.float32).to(device))
    y1 = torch.min(y1_f, torch.tensor((height - 1 + 2 * edge_size)).to(torch.float32).to(device))

    # Calculate indices to draw from flattened version of image batch
    dim2 = (width + 2 * edge_size)
    dim1 = dim2 * (height + 2 * edge_size)
    
    # Set offsets for each image in the batch
    base = dim1 * torch.arange(num_batch).type(tensor_type).to(device)
    base = base.view(-1, 1).repeat(1, height * width).view(-1)
    
    # One pixel shift in Y  direction equals dim2 shift in flattened array
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
            
    idx_00 = (base_y0 + x0)[0].long()
    idx_01 = (base_y0 + x1)[0].long()
    
    idx_10 = (base_y1 + x0)[0].long()
    idx_11 = (base_y1 + x1)[0].long()

    im_flat = im.permute(0,2,3,1).reshape(-1, num_channels)

    # Sample pixels from images
    pix_00 = im_flat[idx_00]
    pix_01 = im_flat[idx_01]
    pix_10 = im_flat[idx_10]
    pix_11 = im_flat[idx_11]

    # Apply linear interpolation to account for fractional offsets
    weight_x0 = x[0] - x0_f[0]
    weight_y0 = y[0] - y0_f[0]
    weight_x1 = x1_f[0] - x[0]
    weight_y1 = y1_f[0] - y[0]

    mul00 = (weight_y1*weight_x1).unsqueeze(dim=1)
    mul01 = (weight_y1*weight_x0).unsqueeze(dim=1)
    mul10 = (weight_y0*weight_x1).unsqueeze(dim=1)
    mul11 = (weight_y0*weight_x0).unsqueeze(dim=1)
    
    weight = pix_00 * mul00 + pix_01 * mul01 + pix_10 * mul10 + pix_11 *mul11

    return weight


def bilinear_sampler_2d(input_images, x_offset, wrap_mode='border', tensor_type = 'torch.FloatTensor', device='cuda'):
    num_batch, num_channels, height, width = input_images.size()

    # Create meshgrid for pixel indicies (PyTorch doesn't have dedicated
    # meshgrid function)
    x = torch.linspace(0, width - 1, width).repeat(height, 1).type(tensor_type).to(device)
    y = torch.linspace(0, height - 1, height).repeat(width, 1).transpose(0, 1).type(tensor_type).to(device)

    # Flatten and repeat for each image in the batch
    x = x.contiguous().view(-1).repeat(1, num_batch)
    y = y.contiguous().view(-1).repeat(1, num_batch)

    x_shift = x_offset[:,0:1,:,:]
    y_shift = x_offset[:,1:2,:,:]

    x_shift = x_shift.contiguous().view(-1)
    y_shift = y_shift.contiguous().view(-1)

    # No scaling
    x = x + x_shift
    y = y + y_shift

    input_transformed = interpolate(input_images, x, y, wrap_mode, tensor_type, device)

    output = input_transformed.reshape(num_batch, height, width, num_channels)

    output = output.permute(0,3,1,2)

    return output