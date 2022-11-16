import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from torchmeta.modules import MetaModule, MetaSequential, DataParallel

class BatchLinear(nn.Linear, MetaModule):
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))

        if self.bias != None:
            bias = params['bias']
            output += bias.unsqueeze(-2)
        
        return output


class MetaSineLayer(MetaModule):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = BatchLinear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input, params=None):
        return torch.sin(self.omega_0 * self.linear(input, params=self.get_subdict(params, 'linear')))


class MetaSirenGrid(MetaModule):
    def __init__(self, args, h_res, w_res, hidden_features, hidden_layers, out_features, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.args = args

        self.h_res = h_res
        self.w_res = w_res
        
        self.net = []
        self.net.append(MetaSineLayer(2, hidden_features, 
                                  is_first=True, omega_0=first_omega_0, bias=args.siren_usebias))

        for i in range(hidden_layers):
            self.net.append(MetaSineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0, bias=args.siren_usebias))

        if outermost_linear:
            final_linear = BatchLinear(hidden_features, out_features, bias=args.siren_usebias)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(MetaSineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0, bias=args.siren_usebias))
        
        self.net = DataParallel(MetaSequential(*self.net))
        print('MLP trainable params:', sum(p.numel() for p in self.net.parameters()))

        self.act = lambda x: x + 0.5
    
    @staticmethod
    def get_mgrid(height, width):
        # Generates a flattened grid of (x,y,...) coordinates in a range of 0 to 1.
        tensors = tuple([torch.linspace(0, 1, steps=t) for t in [height, width]])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        return mgrid
    
    def forward(self, grid=None, params=None):
        if grid == None:
            grid = self.get_mgrid(self.h_res, self.w_res).to(self.args.device)
        
        if params == None:
            params = OrderedDict(self.meta_named_parameters())
        
        grid = grid.clone().unsqueeze(dim=0)

        first_param_key = next(iter(params))
        batch_size = params[first_param_key].shape[0]

        net_in = grid.view(-1, 2)

        output = self.net(net_in, params=self.get_subdict(params, 'net'))

        output = self.act(output)

        output = output.view(batch_size, grid.shape[1], grid.shape[2], -1).permute((0,3,1,2))

        return output
