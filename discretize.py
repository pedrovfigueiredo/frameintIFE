# Loading images
import os
import glob
import cv2
import torch
import numpy as np
import argparse
from util.util import centerCrop
from typing import OrderedDict


from models.meta_siren import MetaSirenGrid
from models.hypernetwork import HyperNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--dir',  type=str, 
                    help='path to input dir', required=True)                    
parser.add_argument("--device", type=str, default="cuda", help="set gpu device")

# Data       
parser.add_argument('--coord_scaler',    type=int,
                    help='number of inner steps',    default = 10)

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

parser.add_argument('--interp_size',    type=int, help='number of generated interpolating images',    default = 5)     


def eval_flows(h_res, w_res, args, hypernet, coords, scaler, model, savedir):
    n_chunks = 4
    grid = model.get_mgrid(height=h_res, width=w_res).to(args.device)
    grids = torch.chunk(grid, n_chunks, dim=1)

    def get_seq(l: float, h: float, rec: int) -> list:
        if rec > 0:
            return get_seq(l, (l+h)/2.0, rec-1) + [(l+h)/2.0] + get_seq((l+h)/2.0, h, rec-1)
        return []
    
    flows = {}
    with torch.no_grad():
        params_4224 = hypernet(coords.clone())

        for t in [0.0] + get_seq(0.0, 1.0, args.interp_size) + [1.0]:
            cur_flows = torch.zeros((1, 2, h_res, 0), device=args.device)
            for grid_i, grid in enumerate(grids):
                params_4224_interp = OrderedDict({k: t*v[0:1] + (1.-t)*v[1:2] for k, v in params_4224.items()})

                cur_flows_tmp = (model(grid=grid, params=params_4224_interp) * 2.0 - 1.0) * scaler
                cur_flows = torch.cat((cur_flows, cur_flows_tmp), dim=-1)
                    
            
            flows["f{:.2f}".format(t)] = cur_flows.cpu().numpy()
    return flows


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    print('Discretizing continous flow representation.')

    img_paths = sorted(glob.glob(os.path.join(args.dir, 'im*.png')))
    h_res,w_res,_ = cv2.imread(img_paths[0]).shape

    weights = torch.load(os.path.join(args.dir, 'weights.ckpt'))

    # Setting up network
    model = MetaSirenGrid(args, w_res=w_res, h_res=h_res, hidden_features=args.siren_feat, hidden_layers=args.siren_depth, out_features=2, 
                                outermost_linear=True, first_omega_0=args.first_omega0, hidden_omega_0=args.hidden_omega0).to(args.device).train()

    hypernet = HyperNetwork(args, hyper_in_features=1, hyper_hidden_layers=args.hyp_depth, hyper_hidden_features=args.hyp_feat, hypo_module=model).to(args.device).train()
    hypernet.load_state_dict(weights['state_dict'])                       

    coordinates = torch.as_tensor([[[[0.0]]], [[[1.0]]]], device=args.device) / args.coord_scaler

    flows = eval_flows(h_res, w_res, args, hypernet, coordinates, 
                  weights['scaler'], model, args.dir)
    
    np.save(os.path.join(args.dir, "flows"), flows)
