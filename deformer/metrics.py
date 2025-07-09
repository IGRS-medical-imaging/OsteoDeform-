import numpy as np
import torch
from torch import nn
import trimesh
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss.chamfer import _handle_pointcloud_input

from .definitions import REDUCTIONS


class ChamferDistKDTree(nn.Module):
    def __init__(self, reduction="mean", njobs=1):
        super(ChamferDistKDTree, self).__init__()
        self.set_reduction_method(reduction)

    def set_reduction_method(self, reduction):
        if not (reduction in list(REDUCTIONS.keys())):
            raise ValueError(
                f"reduction method ({reduction}) not in list of "
                f"accepted values: {list(REDUCTIONS.keys())}"
            )
        self.reduce = REDUCTIONS[reduction]

    def forward(self, src, tar, target_accuracy=False):
        x, x_lengths, _ = _handle_pointcloud_input(src, None, None)
        y, y_lengths, _ = _handle_pointcloud_input(tar, None, None)

        x_nn = knn_points(x, y, lengths1=x_lengths,
                          lengths2=y_lengths, K=1, return_nn=True)
        y_nn = knn_points(y, x, lengths1=y_lengths,
                          lengths2=x_lengths, K=1, return_nn=True)

        src_to_tar_diff = (
            x_nn.knn.squeeze() - src
        )  # [b, m, 3]
        tar_to_src_diff = (
            y_nn.knn.squeeze() - tar
        )  # [b, n, 3]

        accuracy = torch.norm(src_to_tar_diff, dim=-1, keepdim=False)  # [b, m]
        complete = torch.norm(tar_to_src_diff, dim=-1, keepdim=False)  # [b, n]

        chamfer = 0.5 * (self.reduce(accuracy) + self.reduce(complete))
        if not target_accuracy:
            return chamfer  # ,accuracy, complete,
        else:
            return chamfer, self.reduce(complete), self.reduce(accuracy)
