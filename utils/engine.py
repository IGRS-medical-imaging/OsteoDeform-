import time
import torch
import os
import trimesh
from tqdm import tqdm
from deformer.definitions import LOSSES, OPTIMIZERS
from utils.training_utilities import calc_loss, copy_deformer, save_latents
from deformer.metrics import ChamferDistKDTree
import visdom
from utils.visualize_utilities import visualize_with_visdom 
from deformer.latents import LoDPCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import KDTree

# Initialize Visdom with optional parameters
viz = visdom.Visdom(port=8097)

# Create a Visdom environment
env_name = "deformation_training"
viz.env = env_name
win = None

def fine_tuning_with_HFPR(deformer,
               device,
               test_data, 
               args,
               logger,
               loss,
               distance_metric,
               latent_distribution,
               shape_base_instance,
               active_lod=3,
               epoch=60,
               freeze=True):

    (distance_loss, criterion) = loss
    test_mesh, test_point = test_data
    nshapes = len(test_mesh.dataset)
    print(nshapes)

    deformer_test, _ = copy_deformer(
        deformer, args, test_mesh, device, shape_base_instance, nshapes)

    if not freeze:
        deformer_test.module.set_at_layer(len(args.lod))
    else:
        deformer_test.module.set_at_layer(0)

    deformer_test.train()
    unfrozen = active_lod
    epochs = epoch
    lr = .0005

    optim = OPTIMIZERS[args.optim](          
        filter(lambda p: p.requires_grad, deformer_test.module.parameters()), lr=lr)

    logger.info(deformer_test.module)

    with torch.set_grad_enabled(True):
        for epoch in range(1, 1 + epochs):
            total_loss = 0
            count = 0

            for batch_idx, data_tensors in enumerate(test_point):
                tic = time.time()
                data_tensors = [t.to(device) if isinstance(t, torch.Tensor) else t for t in data_tensors]
                (idx, deformed_point_list,  normal_point_list, rescale, normal_path, matched_path) = data_tensors
                mean_rescale = torch.mean(rescale)
                optim.zero_grad()

                target_latents = deformer_test.module.get_lat_params(idx)
                target_latents = latent_distribution.project_data(target_latents)    
                deformed_pts = deformer_test(deformed_point_list, target_latents)

                    
                visualize_with_visdom(viz, deformed_point_list, deformed_pts[-1],
                                      normal_point_list, step=batch_idx)
                
                loss, level_losses = calc_loss(
                    deformed_pts, normal_point_list, distance_loss, criterion)
                loss.backward()
                optim.step()

                logging = ""
                for a, level_loss in enumerate(level_losses):
                    logging += f"Level {a}:{mean_rescale * level_loss} \t"

                deform_abs = torch.mean(torch.norm(deformed_pts[-1] - deformed_point_list, dim=-1))
                toc = time.time()
                dist = level_losses[-1].item()
                bs = len(normal_point_list)
                total_loss += level_losses[-1].detach()
                count += bs

                logger.info(
                    f"Iter: {epoch}, Scaled loss: {loss.item():.4f}\t"
                    f"Dist: {mean_rescale * dist:.4f}\t"
                    f"Deformation Magnitude: {mean_rescale * deform_abs.item():.4f}\t"
                    f"{logging}"
                    f"Time per iter (s): {toc - tic:.4f}\t"
                )

            total_loss /= count
            epoch_loss = total_loss / count
            epoch_losses.append(epoch_loss.item())

            global win
            if win is None:
                win = viz.line(
                    Y=torch.tensor(epoch_losses),
                    X=torch.tensor(range(len(epoch_losses))),
                    opts=dict(title="Epoch Loss", xlabel="Epoch", ylabel="Loss")
                )
            else:
                viz.line(
                    Y=torch.tensor([epoch_loss.item()]),
                    X=torch.tensor([len(epoch_losses) - 1]),
                    win=win,
                    update='append'
                )


    save_latents(deformer_test, test_mesh, args, "test_latents")

    lat_indices = torch.arange(nshapes)
    latents = deformer_test.module.get_lat_params(lat_indices)
    for latent in latents:
        logger.info(
            f"fine tuning latent distribution –– Mean: {latent.flatten().mean()} (+/-{latent.flatten().std()})")

    out_dir = os.path.join(args.log_dir, f"fine_tuning_with_HFPR_test_meshes")
    os.makedirs(out_dir, exist_ok=True)

    cham_dist = evaluate_meshes(
        deformer_test,
        distance_metric,
        test_mesh,
        device,
        criterion,
        logger,
        out_dir,
        latent_distribution=latent_distribution)

    return np.array(cham_dist).mean(), deformer_test


def evaluate_meshes(deformer,
                    distance_metric,
                    test_mesh,
                    device,
                    criterion,
                    logger,
                    out_dir=None,
                    latent_distribution=None
                    ):

    cham_dist = []
    deformations = {}

    deformer.eval()
    with torch.set_grad_enabled(False):
        for ind, data_tensors in enumerate(test_mesh):  # batch size = 1
            idx = torch.tensor([data_tensors[2]], dtype=torch.long)

            # get latent
            target_latents = deformer.module.get_lat_params(idx)
            if latent_distribution:
                target_latents = latent_distribution.project_data(
                    target_latents)

            data_tensors = [t.to(device) if isinstance(t, torch.Tensor) else t for t in data_tensors]
            
            _, _, idx, vi, fi, vj, fj, rescale, normal_file, deformed_file = data_tensors
            v_src = vj  
            f_src = fj  
            v_trg = vi  
            f_trg = fi  

            mean_rescale = torch.mean(rescale)

            vi_j = deformer(
                v_src[..., :3], target_latents,
            )
       
            deformation_dist_norm = vi_j[-1] - v_src[..., :3]
            deformations[deformed_file[0]] = deformation_dist_norm
            deform_abs = torch.mean(torch.norm(
                deformation_dist_norm, dim=-1)) * mean_rescale

            mean_rescale = mean_rescale.detach().cpu().numpy()


            src_vertices = v_src.detach().cpu().numpy()[0] * mean_rescale
            trg_vertices = v_trg.detach().cpu().numpy()[0] * mean_rescale
            deformed_vertices = vi_j[-1].detach().cpu().numpy()[0] * mean_rescale

            # Create PointCloud objects and export as .ply
            src_pc = trimesh.points.PointCloud(src_vertices)
            target_file_name = os.path.basename(deformed_file[0]).split(".")[0]
            trg_pc = trimesh.points.PointCloud(trg_vertices)
            deformed_pc = trimesh.points.PointCloud(deformed_vertices)

            # Save point clouds
            src_pc.export(os.path.join(out_dir, f"{target_file_name}_src.ply"))
            trg_pc.export(os.path.join(out_dir, f"{target_file_name}.ply"))
            deformed_pc.export(os.path.join(out_dir, f"{target_file_name}_deformed.ply"))

            out_path = os.path.join(out_dir, f"{target_file_name}_deformed.ply")
            logger.info(f"Saved to {out_path}")

            _, level_losses = calc_loss(
                vi_j, v_trg, distance_metric["chamfer"], criterion)


            logging = ""
            for a, level_loss in enumerate(level_losses):
                logging += f"Level {a}:{level_loss} \t"

            cham_dist.append(level_losses[-1])

            logger.info(
                f"Test mesh number: {ind}\t"
                f"Chamfer Dist Mean: {level_losses[-1]:.6f}\t"
                f"Deform Mean: {deform_abs.item():.6f}\t"
                f"{logging}"
            )
    cham_dist_np = np.array([cd.cpu().numpy() for cd in cham_dist]) 
    logger.info(
        f"Chamfer Dist Mean: {cham_dist_np.mean()} +/- {cham_dist_np.std()}"
    )
    return cham_dist_np.mean()


def test(deformer,
         device,
         test_data,
         args,
         logger,
         train_data,
         active_lod,
         shapebase_instance,
         epoch=100,
        ):
    """
    Applying for fine tuning with Healthy femur prior Retreival.
    """
    lat_indices = torch.arange(deformer.module.n_shapes)
    latents = deformer.module.get_lat_params(lat_indices)
    print("lat", latents)
    latent_dist = LoDPCA(latents)
    logger.info(type(latent_dist).__name__)

    # Distance metrics
    criterion = LOSSES[args.loss_type]
    chamfer_dist = ChamferDistKDTree(reduction="mean", njobs=1)
    chamfer_dist.to(device)
    chamfer_distance_metric = ChamferDistKDTree(reduction="mean", njobs=1)
    distance_metrics = {"chamfer": chamfer_distance_metric}
    distance_loss = chamfer_dist
    loss = (distance_loss, criterion)

    # Specificity and generality
    fine_tuning_with_HFPR(deformer, device, test_data, args, logger, loss,
                   distance_metrics, latent_dist, shapebase_instance, active_lod=active_lod, epoch=epoch)
epoch_losses = []

def train(args,
    deformer,
    distance_loss,
    dataloader,
    epoch,
    device,
    logger,
    optimizer,
     shape_base_instance
          ):
    """
    Train the OsteoDeform for a single epoch.
    """

    tot_loss = 0
    count = 0
    criterion = LOSSES[args.loss_type]
    toc = time.time()

    deformer.train()

    for batch_idx, data_tensors in enumerate(dataloader):
        tic = time.time()
        data_tensors = [t.to(device) if isinstance(t, torch.Tensor) else t for t in data_tensors]
        (
            _,
            _,
            idx,
            normal_point_list, 
            deformed_point_list, 
            control_points,
            rescale,
            normal_files, 
            deformed_files, 
        ) = data_tensors
        
    
        mean_rescale = torch.mean(rescale)
        bs = len(normal_point_list)
        optimizer.zero_grad()
        print(normal_point_list.shape)
        
        normal_points_batch = normal_point_list
        deformed_points_batch = deformed_point_list

        
        source_points = deformed_points_batch[..., :3]
        target_latents = deformer.module.get_lat_params(idx)

        deformed_pts = deformer(
            source_points, target_latents
        )
        visualize_with_visdom(viz, deformed_points_batch, deformed_pts[-1], normal_points_batch, step=batch_idx)

        loss, level_losses = calc_loss(
            deformed_pts, normal_points_batch, distance_loss, criterion)

        # backprop
        loss.backward()
        optimizer.step()

        logging = ""
        for a, level_loss in enumerate(level_losses):
            logging += "Level {}:{:.6f} \t".format(
                a, mean_rescale * level_loss)

        # Check amount of deformation.
        deform_abs = torch.mean(
            torch.norm(deformed_pts[-1] - deformed_points_batch, dim=-1)
        )

        tot_loss += level_losses[-1].detach()
        count += bs

        # Logger log.
        logger.info(
            f"Train Epoch: {epoch} [{batch_idx * bs}/{len(dataloader) * bs} ({100.0 * batch_idx / len(dataloader):.0f}%)]\t"
            f"Scaled loss: {loss.item():.6f}\t"
            f"Dist Mean: {mean_rescale * loss.item():.6f}\t"
            f"Deform Mean: {mean_rescale * deform_abs.item():.6f}\t"
            f"{logging}"
            f"DataTime: {tic - toc:.4f}\tComputeTime: {time.time() - tic:.4f}\t"

        )
        toc = time.time()

    tot_loss /= count
    epoch_loss = tot_loss / count
    epoch_losses.append(epoch_loss.item())
    global win  
    if win is None:
        win = viz.line(
            Y=torch.tensor(epoch_losses),
            X=torch.tensor(range(len(epoch_losses))),
            opts=dict(title="Epoch Loss", xlabel="Epoch", ylabel="Loss")
        )
    else:
        viz.line(
            Y=torch.tensor([epoch_loss.item()]),
            X=torch.tensor([len(epoch_losses) - 1]),
            win=win,  
            update='append' 
        )
    return tot_loss, epoch_loss, epoch_losses


