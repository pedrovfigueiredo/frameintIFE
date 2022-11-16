# Loading images
import os
import glob
import cv2
import torch
import time
import numpy as np
import argparse

from loss.blendingloss import BlendingLoss
from models.meta_siren import MetaSirenGrid
from models.hypernetwork import HyperNetwork
from util.computeFlowColor import computeImg as flowToColor

from util.xfields_bilinear_sampler import bilinear_sampler_2d
from util.util import RunningAverage

parser = argparse.ArgumentParser()
parser.add_argument('--f_left',  type=str, 
                    help='path to left flow', required=True)                    
parser.add_argument('--f_right',  type=str, 
                    help='path to right flow', required=True)

parser.add_argument('--load_pretrained', type=str, help='loading pretrained model')
parser.add_argument('--savepath', type=str,
                    help='saving path',             default = '')
parser.add_argument("--device", type=str, default="cuda", help="set gpu device")

# Training Options
parser.add_argument('--steps',    type=int,
                    help='number of optimization steps',    default = 10001)                    
parser.add_argument('--lr',       type=float,
                    help='learning rate',          default = 1e-6)
parser.add_argument('--val_freq',    type=int,
                    help='validation frequency',    default = 1000)   
# Siren
parser.add_argument('--first_omega0',    type=int, help='first_omega_0',    default = 10)  
parser.add_argument('--hidden_omega0',    type=int, help='hidden_omega_0',    default = 10)
parser.add_argument('--siren_feat',    type=int, help='number of hidden features on siren net',    default = 128)  
parser.add_argument('--siren_depth',    type=int, help='number of hidden layers on siren net',    default = 5)
parser.add_argument('--siren_usebias', action='store_true', help="use bias on siren")

# Hypernet
parser.add_argument('--hyp_feat',    type=int, help='number of hidden features on siren net',    default = 128)  
parser.add_argument('--hyp_depth',    type=int, help='number of hidden layers on siren net',    default = 1)
parser.add_argument('--hyp_usebias', action='store_true', help="use bias on siren")
parser.add_argument('--coord_scaler',    type=int,
                    help='number of inner steps',    default = 10)

parser.add_argument('--resize_option',
                    choices=['resize', 'none'],
                    default='none',
                    nargs=1,
                    help='Resizing option. Choose between resize and centerCrop') 
parser.add_argument('--target_resolution_factor', type=float, default=1,             
                    help="Resolution in the format: width height")    
parser.add_argument('--save_on_flow_dir', action='store_true', help="if enabled, saves resulting weights on same dir as input flows")                      
parser.add_argument('--n_chunks',    type=int,
                    help='number of chunks',    default = 1)                                                          

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    f_left = torch.permute(torch.load(args.f_left).squeeze(), (1,2,0)).numpy()
    f_right = torch.permute(torch.load(args.f_right).squeeze(), (1,2,0)).numpy()

    if 'resize' in args.resize_option:
        factor = args.target_resolution_factor
        f_left = cv2.resize(f_left, (0,0), fx=factor, fy=factor, interpolation = cv2.INTER_AREA)
        f_right = cv2.resize(f_right, (0,0), fx=factor, fy=factor, interpolation = cv2.INTER_AREA)

    f_left = torch.permute(torch.from_numpy(f_left), (2,0,1)).unsqueeze(dim=0)
    f_right = torch.permute(torch.from_numpy(f_right), (2,0,1)).unsqueeze(dim=0)

    flow_gt = torch.cat((f_left, -f_right), dim=0).to(args.device)
    scaler = torch.max(torch.abs(flow_gt))
    flow_gt = flow_gt / scaler
    # # Adapt to [0,1]
    flow_gt = (flow_gt + 1.0) / 2.0

    _, _, h_res, w_res = flow_gt.shape

    savedir = args.savepath

    l2loss = lambda x,y: torch.mean((x-y)**2)
    blending_loss = BlendingLoss(l2loss, {'reconst' : 1}).to(args.device)

    video_id1 = os.path.basename(os.path.dirname(os.path.dirname(args.f_left)))
    video_id2 = os.path.basename(os.path.dirname(args.f_left))

    if savedir != '':
        savedir = os.path.join(savedir, video_id1, video_id2)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

    # Setting up network
    model = MetaSirenGrid(args, w_res=w_res, h_res=h_res, hidden_features=args.siren_feat, hidden_layers=args.siren_depth, out_features=2, 
                                outermost_linear=True, first_omega_0=args.first_omega0, hidden_omega_0=args.hidden_omega0).to(args.device).train()
    hypernet = HyperNetwork(args, hyper_in_features=1, hyper_hidden_layers=args.hyp_depth, hyper_hidden_features=args.hyp_feat, hypo_module=model).to(args.device).train()
    
    # Loading weights
    if args.load_pretrained:
        print('\n loading pretrained model  {}'.format(args.load_pretrained))
        weights = torch.load(args.load_pretrained)
        hypernet.load_state_dict(weights['state_dict'])

        loading_msg = ''
        if 'epoch' in args.load_pretrained:
            loading_msg = 'Loading at epoch: {}'.format(int(os.path.splitext(args.load_pretrained)[0].split('_')[-1]))
        elif 'iteration' in args.load_pretrained:
            loading_msg = 'Loading at iteration: {}'.format(int(args.load_pretrained.split('.')[0].split('_')[-1]))
        else:
            loading_msg = 'Loading best checkpoint'
            if 'itr' in weights:
                loading_msg += '\n' + 'Loading at iteration: {}'.format(int(weights['itr']))
        
        print(loading_msg)                    

    params = [
        {'params': hypernet.parameters()},
    ]

    optim = torch.optim.Adam(params, lr=args.lr)
    st = time.time()
    coords = torch.as_tensor([[[[0.0]]], [[[1.0]]]], device=args.device) / args.coord_scaler
    coord_chunks = torch.chunk(coords, coords.shape[0])

    n_interpolated_coords = 3

    n_chunks = args.n_chunks
    grid = model.get_mgrid(height=h_res, width=w_res).to(args.device)
    grids = torch.chunk(grid, n_chunks, dim=1)

    im2 = cv2.imread(glob.glob(os.path.join(os.path.dirname(args.f_left), 'im2.*'))[0])
    im2 = np.float32(im2)/255.0
    im2 = torch.from_numpy(np.ascontiguousarray(np.transpose(np.expand_dims(im2, axis=0), (0, 3, 1, 2)))).float().cuda()
    im4 = cv2.imread(glob.glob(os.path.join(os.path.dirname(args.f_left), 'im4.*'))[0])
    im4 = np.float32(im4)/255.0
    im4 = torch.from_numpy(np.ascontiguousarray(np.transpose(np.expand_dims(im4, axis=0), (0, 3, 1, 2)))).float().cuda()

    if savedir != '':
        # Saving gt flow
        for f_i in range(len(flow_gt)):
            gt_f_color = flowToColor(flow_gt[f_i].permute(1,2,0).cpu().detach().numpy())
            cv2.imwrite(os.path.join(savedir, 'gt_flow_t{}.png'.format(f_i)), gt_f_color)

        with torch.no_grad():
            # Computing interpolated estimation
            for i in np.linspace(0,1,n_interpolated_coords+2):
                params_model = hypernet((1.-i)*coord_chunks[0].clone() + i*coord_chunks[1].clone())

                flows_out = (model(params=params_model) * 2.0 - 1.0) * scaler

                for f_i in range(len(flows_out)):
                    single_flow = flows_out[f_i]
                    f_color = flowToColor(single_flow.permute(1,2,0).cpu().detach().numpy())
                    cv2.imwrite(os.path.join(savedir, '{}_s0_t{:.2f}.png'.format('flow', i)), f_color)

                    im4_est = bilinear_sampler_2d(im2, i * flows_out[f_i:f_i+1])
                    im4_est = torch.clamp(im4_est, min=0.0, max=1.0)                                          
                    im4_est = np.transpose(im4_est.cpu().detach().numpy(), (0,2,3,1))[0]*255
                    arguments_strOut = os.path.join(savedir, "interp_s0_img4est_{:.2f}_warpL.png".format(i))
                    cv2.imwrite(arguments_strOut,np.uint8(im4_est))
                    
                    im2_est = bilinear_sampler_2d(im4, -(1-i) * flows_out[f_i:f_i+1])
                    im2_est = torch.clamp(im2_est, min=0.0, max=1.0)                                          
                    im2_est = np.transpose(im2_est.cpu().detach().numpy(), (0,2,3,1))[0]*255
                    arguments_strOut = os.path.join(savedir, "interp_s0_img2est_{:.2f}_warpR.png".format(i))
                    cv2.imwrite(arguments_strOut,np.uint8(im2_est))

    loss_avg = RunningAverage()

    # Optimizing system (inner loop)
    for step in range(args.steps):

        for i in range(coords.shape[0]):
            
            loss_grid = 0.0
            for grid_i, grid in enumerate(grids):
                optim.zero_grad()

                params_model = hypernet(coord_chunks[i].clone())
                flows_out = model(grid=grid, params=params_model)

                loss = torch.tensor(0., device=args.device)
                loss_dict = blending_loss({'out_img': flows_out}, {'reference_img': flow_gt[i:i+1, ..., grid_i*grid.shape[1] : (grid_i+1)*grid.shape[1]]})
                for key, coef in blending_loss.loss_weights.items():
                    value = coef*loss_dict[key]
                    loss += value
                
                loss.backward()
                optim.step()
                loss_grid += loss.item()
            
            loss_avg.update(loss_grid)

        if step % args.val_freq == 0:
            proc_end = time.time()
            time_msg = 'Step {}. Loss: {}. Processing time: {}'.format(step+1, loss_avg(), proc_end - st)
            print(time_msg)

            loss_avg.reset()

            if args.save_on_flow_dir:
                torch.save({'state_dict': hypernet.state_dict(), 'scaler': scaler.item()}, os.path.join(os.path.dirname(args.f_left), 'weights.ckpt'))

            if savedir != '':
                torch.save({'state_dict': hypernet.state_dict(), 'scaler': scaler.item()}, os.path.join(savedir, 'optimized_model.ckpt'))

                    
                with torch.no_grad():
                    # Computing interpolated estimation
                    for i in np.linspace(0,1,n_interpolated_coords+2):
                        params_model = hypernet((1.-i)*coords[0].clone() + i*coords[1].clone())

                        flows_out = (model(params=params_model) * 2.0 - 1.0) * scaler

                        for f_i in range(len(flows_out)):
                            single_flow = flows_out[f_i]
                            f_color = flowToColor(single_flow.permute(1,2,0).cpu().detach().numpy())
                            cv2.imwrite(os.path.join(savedir, '{}_s{}_t{:.2f}.png'.format('flow', step+1, i)), f_color)

                            im4_est = bilinear_sampler_2d(im2, i * flows_out[f_i:f_i+1])
                            im4_est = torch.clamp(im4_est, min=0.0, max=1.0)                                          
                            im4_est = np.transpose(im4_est.cpu().detach().numpy(), (0,2,3,1))[0]*255
                            arguments_strOut = os.path.join(savedir, "interp_s{}_img4est_{:.2f}_warpL.png".format(step+1, i))
                            cv2.imwrite(arguments_strOut,np.uint8(im4_est))
                            
                            im2_est = bilinear_sampler_2d(im4, -(1-i) * flows_out[f_i:f_i+1])
                            im2_est = torch.clamp(im2_est, min=0.0, max=1.0)                                          
                            im2_est = np.transpose(im2_est.cpu().detach().numpy(), (0,2,3,1))[0]*255
                            arguments_strOut = os.path.join(savedir, "interp_s{}_img2est_{:.2f}_warpR.png".format(step+1, i))
                            cv2.imwrite(arguments_strOut,np.uint8(im2_est))
