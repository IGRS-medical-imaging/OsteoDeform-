import argparse
import os
import glob
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from deformer.definitions import LOSSES, OPTIMIZERS, SOLVERS
from deformer.metrics import ChamferDistKDTree
from utils.training_utilities import initialize_environment, get_deformer, save_latents
from utils.utility_functions import snapshot_files, save_checkpoint
from utils.engine import train, test, evaluate_meshes, new_test
import utils.data_loader_file as dl

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--increase_layer",
        type=int,
        default=None,
        help="Add layer of detail every x epochs",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="RBF epsilon",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="R",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--lod",
        type=int,
        nargs='+',
        default=None,
        help="Dimensions of the level of detail for the embeddings. (defaul: None)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="path to mesh folder root",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=SOLVERS,
        default="dopri5",
        help="ode solver. (default: dopri5)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="absolute error tolerence in ode solver. (default: 1e-5)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="relative error tolerence in ode solver. (default: 1e-5)",
    )
    parser.add_argument(
        "--log_dir", type=str, required=True, help="log directory for run"
    )
    parser.add_argument(
        "--nonlin", type=str, default="elu", help="type of nonlinearity to use"
    )
    parser.add_argument(
        "--optim", type=str, default="adam", choices=list(OPTIMIZERS.keys()), help="type of optimizer to use"
    )
    parser.add_argument(
        "--loss_type", type=str, default="l2", choices=list(LOSSES.keys()),
        help="type of loss to use"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=False, # Changed this line
        help="path to checkpoint if resume is needed",
    )
    parser.add_argument(
        "--lat_dims", default=32, type=int, help="number of latent dimensions."
    )
    parser.add_argument(
        "--deformer_nf",
        default=100,
        type=int,
        help="number of base number of feature layers in deformer (imnet).",
    )
    parser.add_argument(
        "--lr_scheduler", dest="lr_scheduler", action="store_true", default=False
    )
    parser.add_argument(
        '--independent_epsilon', action='store_true', default=False
    )
    parser.add_argument(
        "--adjoint",
        dest="adjoint",
        action="store_true",
        help="use adjoint solver to propagate gradients thru odeint.",
    )
    parser.add_argument(
        "--no_adjoint",
        dest="adjoint",
        action="store_false",
        help="not use adjoint solver to propagate gradients thru odeint.",
    )
    parser.add_argument(
        "--sample_surface",
        action="store_true",
        default=True,
        help="Sample surface points.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,  # You can change this default value
        help="Number of samples to process per mesh (default: 1024).",
    )
    parser.add_argument(
        '--rbf', default=False, action='store_true'
    )
    parser.set_defaults(lr_scheduler=True)
    parser.set_defaults(adjoint=True)
    parser.set_defaults(vis_mesh=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    args, kwargs, device, logger = initialize_environment(args)
    # Log and create snapshots.
    filenames_to_snapshot = (
        glob.glob("*.py") + glob.glob("*.sh") +
        glob.glob("deformer/*.py")
    )
    snapshot_files(filenames_to_snapshot, args.log_dir)
    
    shape_base_instance, _, train_mesh, train_points, val_mesh, val_points, test_mesh, _, finetuning_with_HFPR= dl.build(
        args, kwargs, logger)
    
    deformer = get_deformer(args, train_mesh, device, shape_base_instance, len(train_mesh.dataset))

    all_model_params = list(deformer.parameters())
    optimizer = OPTIMIZERS[args.optim](all_model_params, lr=args.lr)
         
    # Initialize losses
    chamfer_dist = ChamferDistKDTree(reduction="mean", njobs=1)
    chamfer_dist.to(device)
    distance_loss = chamfer_dist

    # Set variables for training
    start_ep = 0
    global_step = np.zeros(1, dtype=np.uint32)
    tracked_stats = np.inf

    # Resume training
    if args.resume:
        logger.info(
            "Loading checkpoint {} ================>".format(args.resume)
        )
        resume_dict = torch.load(args.resume)
        start_ep = resume_dict["epoch"]
        global_step = resume_dict["global_step"]
        tracked_stats = resume_dict["tracked_stats"]
        deformer.load_state_dict(resume_dict["deformer_state_dict"])
        logger.info("[!] Successfully loaded checkpoint.")
        
    deformer.to(device)
    deformer = nn.DataParallel(deformer)
    
    deformer.float()

    # Log model description
    logger.info(deformer.module)
    logger.info(deformer)

    checkpoint_path = os.path.join(args.log_dir, "checkpoint_latest_")

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    unfrozen = len(args.lod)
    
    
    test(deformer=deformer,
         device=device,
         test_data= (test_mesh, finetuning_with_HFPR),
         args=args,
         logger=logger,
         train_data=train_mesh,
         active_lod=len(args.lod), # hardcoding to len(args.lod)
         shapebase_instance= shape_base_instance,
         epoch=100
         )
    


if __name__ == "__main__":
    main()













