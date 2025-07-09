import json
import copy
import numpy as np
import os
import trimesh
import torch
import torch.optim as optim
from utils.utility_functions import get_logger
import point_cloud_utils as pcu
from torch import nn
from deformer.deformer_main import NeuralFlowDeformer
from deformer.definitions import OPTIMIZERS

def initialize_environment(args): 
    os.makedirs(args.log_dir, exist_ok=True)
    if int(torch.cuda.device_count()):
        args.batch_size = args.batch_size
    else:
        args.batch_size = 1
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    kwargs = (
        {"num_workers": min(6, args.batch_size), "pin_memory": True}
        if use_cuda
        else {}
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    args.at_layer = 0
    logger = get_logger(log_dir=args.log_dir)
    with open(os.path.join(args.log_dir, "params.json"), "w") as fh:
        json.dump(args.__dict__, fh, indent=2)
    logger.info("%s", repr(args))
    args.n_vis = 10
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return args, kwargs, device, logger


def copy_deformer(deformer, args, train_data, device, n_shapes):
    
    deformer_test = get_deformer(args, train_data, None)
    deformer_test = nn.DataParallel(deformer_test)

    model_dict = deformer_test.state_dict()
    pretrained_dict = copy.deepcopy(deformer.state_dict())

    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    deformer_test.load_state_dict(model_dict)

    for param in deformer_test.module.parameters():
        param.requires_grad = False

    embedded_latents = get_latents(args, n_shapes, 0.0001)
    embedded_latents.to(device)

    deformer_test.module.add_lat_params(embedded_latents)
    deformer_test.to(device)

    return deformer_test, embedded_latents


def get_deformer(args, dataloader, device, shape_base_instance, n_shapes):
    mean_shape = r"/home/imaging/new_nfd_03_02_2025/mean_shape_all_data.ply" # provide the file path for the mean shape of the osteophytic femurs 
    mesh = trimesh.load(mean_shape)
    points = mesh.vertices
    sampled_points = points[np.random.choice(points.shape[0], 800, replace=False)]
    grid_tensor = torch.from_numpy(sampled_points).float()
    control_points = grid_tensor
    # Initialize the RBF 
    args.rbf = (
                shape_base_instance,
                control_points,
                args.lod,
                None,
                args.independent_epsilon)

    deformer = NeuralFlowDeformer(
        latent_size=args.lat_dims,
        dim=0 if args.fourier else 3,
        device = device,
        f_width=args.deformer_nf,
        method=args.solver,
        nonlinearity=args.nonlin,
        adjoint=args.adjoint,
        rtol=args.rtol,
        atol=args.atol,
        lod=args.lod,
        at_layer=args.at_layer,
        rbf=args.rbf,  
    )
    if n_shapes is not None:
        lat_params = get_latents(args, n_shapes, device)
        deformer.add_lat_params(lat_params)
    return deformer

def get_latents(args, n_shapes, device, magnitude=0.1): 
    lat_params = torch.nn.ParameterList([])
    for i in args.lod:
        lat_tensor = torch.randn(n_shapes, args.lat_dims, i, i, i, device=device)
        lat_tensor = lat_tensor * magnitude
        lat_params.append(torch.nn.Parameter(lat_tensor, requires_grad=True))
    return lat_params

def save_latents(deformer, dataloader, args, name="training_latents"):
    """Save latents with filename.
    """
    latents = {}
    lat_indices = torch.arange(len(dataloader.dataset))
    for idx in lat_indices:
        normal_file, deformed_file = dataloader.dataset.idx_to_file_mapping[idx.item()]
        latents[deformed_file] = deformer.module.get_lat_params(idx)
    torch.save(latents, os.path.join(args.log_dir, f"{name}.pkl"))
        

def calc_loss(deformations, targets, distance, criterion):
    level_losses = []
    for i in range(len(deformations)):
        dist = distance(deformations[i], targets)
        # average over batch
        level_losses.append(dist.mean(0))
        if i == 0:
            distances = dist
        else:
            distances = dist
    if torch.is_tensor(dist):
        loss = criterion(distances, torch.zeros_like(distances))
    else:
        loss = distances

    return loss, level_losses
