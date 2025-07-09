"""
Credits:
Initial implementation by David Ludke and Tamaz Amiranashvili
https://github.com/davecasp/flowssm.git

Novel application, modified classes and functions.
"""


import torch
from torch import nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_regular
import numpy as np
import point_cloud_utils as pcu
from deformer.definitions import NONLINEARITIES
from utils.data_loader_file import ShapeBase, rescale_center
import os 

torch.pi = torch.acos(torch.zeros(1)).item() * 2

class RadialBasisFunctionSampler3D(nn.Module):
    """
    enables the mapping of latent features from a set of 3D control points 
    to the input osteophytic femur point cloud
    """
    def __init__(self,
                 shape_base = ShapeBase,
                 control_points = None,
                 n_gridpoints=100,
                 extent=None,
                 independent_epsilon=False,
                 epsilon=0.5):
        super(RadialBasisFunctionSampler3D, self).__init__()

       
        self.shape_base = shape_base  
        self.extent_dict = self.shape_base.extent_dict
        self.center_dict = self.shape_base.center_dict
        self.grid_dim = n_gridpoints  
        n_gridpoints = n_gridpoints**3  # Number of control points
        epsilon = epsilon * self.grid_dim

        # Initialize epsilon for the RBF kernel
        if independent_epsilon:
            self.n_epsilon = n_gridpoints
            self.epsilon = nn.Parameter(
                torch.ones((self.n_epsilon)) * epsilon)  # Trainable epsilon
        else:
            self.n_epsilon = 1
            self.register_buffer(
    'epsilon', 
    torch.ones((self.n_epsilon), device="cuda:0") * epsilon 
)
        # Initialize the control points
        self.grid = self.get_controll_points(n_gridpoints,  control_points)
        # RBF interpolation flag
        self.interpolation = n_gridpoints != 1
        self.n_grid = len(self.grid)

    def forward(self, points, latents): 
        B, N, _ = points.shape  
        latents = latents.reshape((B, latents.shape[-1]**3, latents.shape[1])) 
        latents = latents[:, :self.n_grid, :] 
        if not self.interpolation:
            return latents.repeat((1, N, 1))
        else:
            with torch.no_grad():
                dist = self.distance(points) 
            data = self.rb_function(dist) 
            return torch.bmm(data, latents)

    def distance(self, points):
        return torch.cdist(points, self.grid) 

    def rb_function(self, data):
        return torch.exp(-((self.epsilon * data)**2))  # Gaussian Kernel

    def get_controll_points(self, n_gridpoints, control_points):
        with torch.no_grad():
            if n_gridpoints == 1:
                points = torch.tensor([0])
                x, y, z = torch.meshgrid(
                    points, points, points, indexing="ij")
                grid = torch.cat((x.flatten()[:, None], y.flatten()[
                                    :, None], z.flatten()[:, None]), dim=1).float()
                grid = grid.to('cuda:0')
            else:
                grid = control_points # [grid_points, 3]
                print("length of grid", len(grid))
                grid = grid.to('cuda:0')
        return grid                 

    def set_epsilon(self, epsilon):
        """Set epsilon value.
        """
        self.epsilon = torch.nn.Parameter(torch.ones((self.n_epsilon),
                                                     device=self.epsilon.device) *
                                          epsilon * self.grid_dim)

class ImNet(nn.Module):
    """
    parametrizes the velocity field
    """
    def __init__(
        self,
        dim=3,
        in_features=32,
        out_features=4,
        nf=32,
        nonlinearity="leakyrelu",
    ):
        super(ImNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        self.dimz = dim + in_features
        self.out_features = out_features 
        self.nf = nf
        self.activ = NONLINEARITIES[nonlinearity]
        self.fc0 = nn.Linear(self.dimz, nf * 16) 
        self.fc1 = nn.Linear(nf * 16 + self.dimz, nf * 8) 
        self.fc2 = nn.Linear(nf * 8 + self.dimz, nf * 4) 
        self.fc3 = nn.Linear(nf * 4 + self.dimz, nf * 2) 
        self.fc4 = nn.Linear(nf * 2 + self.dimz, nf * 1) 
        self.fc5 = nn.Linear(nf * 1, out_features)
        self.fc = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
        self.fc = nn.ModuleList(self.fc)

    def forward(self, x):
        x_tmp = x
        for dense in self.fc[:4]:
            x_tmp = self.activ(dense(x_tmp))
            x_tmp = torch.cat([x_tmp, x], dim=-1)
        x_tmp = self.activ(self.fc4(x_tmp))
        x_tmp = self.fc5(x_tmp)
        return x_tmp  


class NeuralFlowModel(nn.Module):
    """
        With the corresponding point and the interpolated latent vectors as inputs this 
        outputs the velocity vector at each point
    """
    def __init__(
        self,
        dim=3,
        device=None, 
        out_features=3,
        latent_size=1,
        f_width=50,
        nonlinearity="relu",
    ):
        super(NeuralFlowModel, self).__init__()

        self.out_features = out_features
        self.net = ImNet(
            dim=dim,
            in_features=latent_size,
            out_features=out_features,
            nf=f_width,
            nonlinearity=nonlinearity,
        )  
        self.net = self.net.to(device=device)

        self.latent_updated = False
        self.rbf = None
        self.lat_params = None
        self.scale = nn.Parameter(torch.ones(1, device=device) * 1e-3)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)
                nn.init.constant_(m.bias, val=0)

    def add_lat_params(self, lat_params):
        """
        Add latent parameters.
        """
        self.lat_params = lat_params

    def add_rbf(self, rbf):
        """
        Add radial basis function interpolation.
        """
        self.rbf = rbf

    def get_lat_params(self, idx):
        """
        Get latent parameters for indices.
        """
        assert self.lat_params is not None
        return self.lat_params[idx]

    def get_rbf(self):
        """
        Get radial basis function interpolator.
        """
        return self.rbf

    def update_latents(self, latent_sequence): 
        """
        Save latents for ODEINT solve and compute vectornorm on latent.  
        """
        self.latent_sequence = latent_sequence
        self.latent_norm = torch.norm(
            self.latent_sequence, dim=-1
        )
        self.latent_updated = True

    def latent_at_t(self, t): 
        latent_val = t * self.latent_sequence 
        return latent_val

    def get_velocity(self, latent_vector, points): 
        points_latents = torch.cat((points, latent_vector), axis=-1)
        b, n, d = points_latents.shape
        out = self.net(points_latents.reshape([-1, d]))
        out = out.reshape([b, n, self.out_features])
        return out

    def forward(self, t, points):
        if not self.latent_updated:
            raise RuntimeError(
                "Latent not updated. "
                "Use .update_latents() to update the source and target latents"
            )

        t = t.to(self.latent_sequence.device)
        latent_val = self.latent_at_t(t)
        velocity = self.get_velocity(
            latent_val, points)  
        velocity *= self.latent_norm[:, :, None]
        return velocity * self.scale


class NeuralFlowDeformer(nn.Module):
    
    """ deformation formulated as the solution to an ordinary differential equation (ODE)
        by integrating the velocity field conditioned on the latent codes
    """

    def __init__(
        self,
        dim=3,
        device=None,
        out_features=3,
        latent_size=32,
        f_width=50,
        method="dopri5",
        nonlinearity="leakyrelu",
        adjoint=True,
        atol=1e-5,
        rtol=1e-5,
        lod=None,
        at_layer=0,
        rbf=None,
    ):

        super(NeuralFlowDeformer, self).__init__()
        self.lod = lod
        self.at_layer = at_layer
        self.device = device

        # ODEINT
        self.method = method
        self.adjoint = adjoint
        self.rtol = rtol
        self.atol = atol
        self.odeints = [
            odeint_adjoint if adjoint else odeint_regular] * len(self.lod)
        self.__timing = torch.from_numpy(
            np.array([0.0, 1.0]).astype("float32")
        ).float()

        # Neural Flow Model
        self.out_features = out_features
        self.nonlinearity = nonlinearity
        self.f_width = f_width
        self.rbf = rbf
        self.latent_size = latent_size

        self.net = torch.nn.ModuleList([])
        for i in range(len(self.lod)):
                self.net.append(NeuralFlowModel(
                    dim=dim,
                    device=device,
                    out_features=out_features,
                    latent_size=latent_size,
                    f_width=f_width,
                    nonlinearity=nonlinearity
                ))
        (
                 shape_base ,
                 control_points, 
                 n_grid_p,
                 extent,
                 independent_epsilon,
                 ) = rbf
        for i in range(len(self.lod)):
                self.net[i].add_rbf(RadialBasisFunctionSampler3D(
                 shape_base,
                 control_points,
                 n_grid_p[i],
                 extent,
                 independent_epsilon,
                 ))

        self.lasts = [parameter.data for name, 
                      parameter in self.net[0].named_parameters() if "fc5" in name]

        # init gradients for layers
        for i in range(min(self.at_layer, len(self.net))):
            self.set_latent_gradient(i)
            self.set_layer_gradient(i)

    @ property
    def adjoint(self):
        return self.__adjoint

    @ adjoint.setter
    def adjoint(self, isadjoint):
        assert isinstance(isadjoint, bool)
        self.__adjoint = isadjoint

    @ property
    def timing(self):
        return self.__timing

    @ property
    def n_shapes(self):
        n = self.net[0].lat_params.shape[0]
        return n

    @ timing.setter
    def timing(self, timing):
        assert isinstance(timing, torch.tensor)
        assert timing.ndim == 1
        self.__timing = timing

    def add_lat_params(self, lat_params):
        for i in range(len(self.lod)):
            self.net[i].add_lat_params(lat_params[i])

    def get_lat_params(self, idx):
        latents = []
        for i in range(len(self.lod)):
            latents.append(self.net[i].get_lat_params(idx))
        return latents

    def set_at_layer(self, layer_id):
        self.at_layer = layer_id

    def set_latent_gradient(self, idx, freeze=True):
        for name, param in self.net[idx].named_parameters():
            if 'lat' in name:
                param.requires_grad = freeze

    def set_layer_gradient(self, idx, freeze=True):
        for name, param in self.net[idx].named_parameters():
            if 'lat' not in name:
                param.requires_grad = freeze
    
    def update_latents(self, latent_sequence):

        self.latent_sequence = latent_sequence
        self.latent_norm = torch.norm(
            self.latent_sequence, dim=-1
        )
        self.latent_updated = True

    def __str__(self):
        string = "\n\n################################################################################################\n"
        string += "######################################### Model Summary ########################################\n"
        string += "################################################################################################\n"
        string += "\n\nPyTorch implementation of a continuous flow-field deformer.\n\n"
        string += "------------------------------------------------------------------------------------------------\n"
        string += "-------------------------------------------Model set-up-----------------------------------------\n"
        string += "------------------------------------------------------------------------------------------------\n"
        string += f"{'3D' if self.out_features == 3 else '2D'} imnet deformers without sign network, that "
        string += f"applies template deformations\n"

        string += f"It leverages {len(self.net)} sequential deformation(s), with lod resolution(s) of {['{}x{}x{}'.format(l.shape[3],l.shape[3],l.shape[3]) for l in self.get_lat_params(0)]}\n"

        string += "------------------------------------------------------------------------------------------------\n"
        string += "--------------------------------------------MLP set-up------------------------------------------\n"
        string += "------------------------------------------------------------------------------------------------\n"

        string += f"6 MLP layers with width of {self.f_width}\n"
        string += f"{self.latent_size} latent features\n"
        string += f"{self.nonlinearity} activation function\n"
        string += "------------------------------------------------------------------------------------------------\n"
        string += "---------------------------------------------Odeint---------------------------------------------\n"
        string += "------------------------------------------------------------------------------------------------\n"
        string += f"{self.method} odeint method {'with' if self.adjoint else 'without'} adjoint (absolute tolerance {self.atol}, relative tolerance {self.rtol})\n\n"

        string += "------------------------------------------------------------------------------------------------\n"
        string += "-------------------------------------------Parameters-------------------------------------------\n"
        string += "------------------------------------------------------------------------------------------------\n"

        for n, deformer in enumerate(self.net):
            string += f"Deformer {n} with {sum(p.numel() for name, p in deformer.named_parameters() if p.requires_grad)} trainable parameters\n"
            string += f"Deformer {n} with {sum(p.numel() for name, p in deformer.named_parameters() if p.requires_grad and 'lat_params' in name)} trainable ({sum(p.numel() for name, p in deformer.named_parameters() if 'lat_params' in name)}) latent features\n"
        if self.rbf:
            for net in self.net:
                string += f"RBF epsilon of {net.get_rbf().epsilon}.\n"

        string += "################################################################################################\n"
        string += "################################################################################################\n"
        string += "\n\n"
        return string

    def forward(self, points, latent_sequence):
        timing = self.timing.to(points.device)
        deformations = []
        targets = []
        points_sample = points

        # Iterate over level of detail
        for i in range(len(latent_sequence)):
                ##################
                ## MLP DEFORMER ##
                ##################
                targets.append(self.net[i].get_rbf()(
                        points_sample,
                        latent_sequence[i]
                    ))

                target = targets[-1]
                _latent_sequence = target
                _ = self.net[i].update_latents(_latent_sequence)

                # solve ODEINT
                points_deformed = self.odeints[i](
                    self.net[i],
                    points,
                    timing,
                    method=self.method,
                    rtol=self.rtol,
                    atol=self.atol,
                )
                deformations.append(points_deformed[-1])
                points = points_deformed[-1]
        return deformations
