import glob
import numpy as np
import os
import torch
import trimesh
import open3d as o3d
import point_cloud_utils as pcu
from torch.utils.data import Dataset, DataLoader
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from scipy.spatial import KDTree

def rescale_center(points, extent=None, center=None):
    if torch.is_tensor(center):
        center = center.cpu().detach().numpy()
    if not extent:
        extent = (max(np.max(points, axis=0) - np.min(points, axis=0))) / 2
    # Scale
    points /= extent
    # Center
    if center is None:
        center = (np.max(points, axis=0) - np.min(points, axis=0)) / \
            2 - np.max(points, axis=0)
    points += center
    return points, torch.tensor(extent), torch.tensor(center)

class ShapeBase(Dataset):
    def __init__(self, data_root, split, center_and_rescale=True, center=None, extent=None):
        self.data_root = data_root
        self.split = split
        self.center_and_rescale = center_and_rescale
        self.normal_files, self.deformed_files = self._get_filenames(data_root, split)
        self.extent_dict = {}
        self.center_dict = {}
        if self.center_and_rescale:
            self._compute_rescaling_and_centering()
        self._file_splits = None
        self._fname_to_idx_dict = None
        self._idx_to_file_mapping = None  # New mapping dictionary
        self._file_to_idx_mapping = None

    def _compute_rescaling_and_centering(self):
        for normal_file in self.normal_files:
            base_name = os.path.splitext(os.path.basename(normal_file))[0]
            print(normal_file)
            normal_vertices = trimesh.load(normal_file, process=False).vertices
            _, extent, center = rescale_center(normal_vertices)
            self.extent_dict[base_name] = extent.item()
            self.center_dict[base_name] = center.numpy()

    @property
    def n_shapes(self):
        return len(self.normal_files)

    @property
    def file_splits(self):
        if self._file_splits is None:
            self._file_splits = {"train": [], "test": [], "val": []}
            for normal_file in self.normal_files:
                normalized_path = os.path.normpath(normal_file)
                if "train" in normalized_path.split(os.sep):
                    self._file_splits["train"].append(normal_file)
                elif "test" in normalized_path.split(os.sep):
                    self._file_splits["test"].append(normal_file)
                elif "val" in normalized_path.split(os.sep):
                    self._file_splits["val"].append(normal_file)
                else:
                    print(f"Warning: File '{normal_file}' does not belong to train/test/val")
        return self._file_splits


    @staticmethod
    def _get_filenames(data_root, split):
        normal_folder = os.path.join(data_root, split, "normal")
        deformed_folder = os.path.join(data_root, split, "deformed")
        normal_files = sorted(glob.glob(os.path.join(normal_folder, "*.ply")))
        deformed_files = {}
        for normal_file in normal_files:
            base_name = os.path.splitext(os.path.basename(normal_file))[0]
            deformed_subfolder = os.path.join(deformed_folder, base_name)
            if os.path.exists(deformed_subfolder):
                deformed_files[base_name] = sorted(glob.glob(os.path.join(deformed_subfolder, "*.ply")))
        return normal_files, deformed_files

    @property
    def idx_to_file_mapping(self):
        """Create a mapping from idx to the corresponding normal and deformed file pair."""
        if self._idx_to_file_mapping is None:
            idx_to_file_mapping = {}
            for i, normal_file in enumerate(self.normal_files):
                base_name = os.path.splitext(os.path.basename(normal_file))[0]
                deformed_file_list = self.deformed_files.get(base_name, [])
                for j, deformed_file in enumerate(deformed_file_list):
                    idx = self.combinations_to_idx(i, j)
                    idx_to_file_mapping[idx] = (normal_file, deformed_file)
            self._idx_to_file_mapping = idx_to_file_mapping
        return self._idx_to_file_mapping

    def file_to_idx_mapping(self):
        """Create a mapping from a (normal_file, deformed_file) pair to the corresponding idx."""
        if self._file_to_idx_mapping is None:
            file_to_idx_mapping = {}
            for i, normal_file in enumerate(self.normal_files):
                base_name = os.path.splitext(os.path.basename(normal_file))[0]
                deformed_file_list = self.deformed_files.get(base_name, [])
                for j, deformed_file in enumerate(deformed_file_list):
                    idx = self.combinations_to_idx(i, j)
                    file_to_idx_mapping[(normal_file, deformed_file)] = idx
            self._file_to_idx_mapping = file_to_idx_mapping
        return self._file_to_idx_mapping
    
    def get_idx_from_file(self, file_path):
        """
        Given a file path (normal or deformed), return the corresponding idx.
        """
        # Ensure the mapping is built
        file_to_idx = self.file_to_idx_mapping()

        for (normal_file, deformed_file), idx in file_to_idx.items():
            if file_path == normal_file or file_path == deformed_file:
                return idx

        raise ValueError(f"File not found in mapping: {file_path}")


    def idx_to_combinations(self, idx):
        """Convert index to a normal file and its deformed pair."""
        i = idx //  9 # Each normal file has 9 deformed variations
        j = idx %   9# Deformed variation for the normal file
        if hasattr(idx, "__len__"):
            i = np.array(i, dtype=int)
            j = np.array(j, dtype=int)
        else:
            i = int(i)
            j = int(j)
        return i, j

    def combinations_to_idx(self, i, j):
        """Convert normal-deformed pair indices to a single index."""
        idx = i * 9 + j  # Each normal file has 9 deformed variations
        if hasattr(idx, "__len__"):
            idx = np.array(idx, dtype=int)
        else:
            idx = int(idx)
        return idx

    def __len__(self):
        """Total number of deformed shapes."""
        return sum(len(files) for files in self.deformed_files.values())

    def __getitem__(self, idx):
        normal_file, deformed_file = self.idx_to_file_mapping[idx]
        base_name = os.path.splitext(os.path.basename(normal_file))[0]
        extent = self.extent_dict[base_name]
        center = self.center_dict[base_name]
        return normal_file, deformed_file, extent, center
    
class ShapePointset(ShapeBase):
    def __init__(
        self,
        data_root,
        split,
        nsamples=None,
        n_gridpoints = None,
        center_and_rescale=True,
        center=None,
        extent=None
    ):
        super(ShapePointset, self).__init__(
            data_root=data_root,
            split=split,
            center_and_rescale=center_and_rescale,
            center=center,
            extent=extent
        )
        self.nsamples = nsamples
        self.n_gridpoints = n_gridpoints

    @staticmethod
    
    def poisson_sampling(deformed_file, n_gridpoints, extent, center):
        v, f, n = pcu.load_mesh_vfn(deformed_file)  # Vertices, faces, normals
        v, _, _ = rescale_center(v, extent, center)
        f_i, bc = pcu.sample_mesh_poisson_disk(v, f, n_gridpoints)  # Returns faces and barycentric coordinates
        grid = pcu.interpolate_barycentric_coords(f, f_i, bc, v)  # Interpolate sampled points
        grid_tensor = torch.from_numpy(grid).float()  # Shape: [grid_points, 3]
        return grid_tensor
        
    def sample_points(self, mesh_path, nsamples):
        mesh = trimesh.load(mesh_path, process=False)
        print(mesh_path)
        verts = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        mesh = Meshes(verts=[verts], faces=[faces])
        sample_points = sample_points_from_meshes(mesh, nsamples)
        sample = sample_points.squeeze().cpu().numpy()
        return sample


    def _get_one_mesh(self, idx, n_gridpoints, use_deformed=False):
        normal_file, deformed_file = self.idx_to_file_mapping[idx]
        file_to_sample = deformed_file if use_deformed else normal_file
        base_name = os.path.splitext(os.path.basename(normal_file))[0]
        verts = self.sample_points(file_to_sample, self.nsamples)
        extent = self.extent_dict[base_name]
        center = self.center_dict[base_name]

        if self.center_and_rescale:
            verts, target_scale, _ = rescale_center(verts, extent, center)
        else:
            target_scale = torch.Tensor([1])  # No rescaling needed

        grid = self.poisson_sampling(file_to_sample, n_gridpoints, extent, center)
        verts = torch.from_numpy(verts).float()
        return verts, target_scale, grid

    def __getitem__(self, idx):
        i, j = self.idx_to_combinations(idx)
        verts_i, _, grid = self._get_one_mesh(idx,self.n_gridpoints, use_deformed=False)
        verts_j, target_scale, _= self._get_one_mesh(idx,self.n_gridpoints, use_deformed=True)
        idx = self.combinations_to_idx(i,j)
        normal_file, deformed_file = self.idx_to_file_mapping[idx]
        return i, j, idx, verts_i, verts_j, grid, target_scale, normal_file, deformed_file
    

class ShapeVertex(ShapeBase):   
    """Pytorch Dataset for sampling vertices from meshes."""

    def __init__(
        self,
        data_root,
        split,
        nsamples=None,
        center_and_rescale=True,
        center=None,
        extent=None, 
    ):
        super(ShapeVertex, self).__init__(
            data_root=data_root,
            split=split,
            center_and_rescale=center_and_rescale,
            center=center,
            extent=extent
        )
        self.nsamples = nsamples

    @staticmethod
    def sample_mesh(mesh_path, nsamples):
        mesh = trimesh.load(mesh_path, process=False)
        v = np.array(mesh.vertices)
        seq = np.random.permutation(len(v))[:nsamples]
        if len(seq) < nsamples:
            seq_repeat = np.random.choice(
                len(v), nsamples - len(seq), replace=True)
            seq = np.concatenate([seq, seq_repeat], axis=0)
            v = v[seq]
        return v

    def _get_one_mesh(self, idx, use_deformed=False):
        normal_file, deformed_file = self.idx_to_file_mapping[idx]
        file_to_sample = deformed_file if use_deformed else normal_file
        base_name = os.path.splitext(os.path.basename(normal_file))[0]
        verts = self.sample_mesh(file_to_sample, self.nsamples)
        extent = self.extent_dict[base_name]
        center = self.center_dict[base_name]
        if self.center_and_rescale:
            verts, target_scale, _ = rescale_center(verts, extent, center)
        else:
            target_scale = torch.Tensor([1]) 
        verts = torch.from_numpy(verts).float()
        return verts, target_scale

    def __getitem__(self, idx):
        i, j = self.idx_to_combinations(idx)
        verts_i, _ = self._get_one_mesh(idx, use_deformed=False)
        verts_j, target_scale = self._get_one_mesh(idx, use_deformed=True)
        return i, j, verts_i, verts_j, target_scale

class ShapeMesh(ShapeBase):

    def __init__(self,
                 data_root,
                 split,
                 nsamples,
                 center_and_rescale=True,
                 center=None,
                 extent=None):
        
        super(ShapeMesh, self).__init__(
            data_root=data_root,
            split=split,
            center_and_rescale=center_and_rescale,
            extent=extent,
            center=center)
        
        self.nsamples = nsamples

    def get_pairs(self, idx):
        verts_i, faces_i, _ = self.get_single(idx, use_deformed=False)
        verts_j, faces_j, target_scale = self.get_single(idx, use_deformed=True)
        i,j = self.idx_to_combinations(idx)
        normal_file, deformed_file = self.idx_to_file_mapping[idx]
        verts_i_np = verts_i.numpy()  # Deformed femur vertices
        verts_j_np = verts_j.numpy()  # Normal femur vertices

        # Create Open3D point clouds
        point_cloud_i = o3d.geometry.PointCloud()
        point_cloud_j = o3d.geometry.PointCloud()
        point_cloud_i.points = o3d.utility.Vector3dVector(verts_i_np)
        point_cloud_j.points = o3d.utility.Vector3dVector(verts_j_np)

        return i, j, idx, verts_i, faces_i, verts_j, faces_j, target_scale, normal_file, deformed_file

    def get_single(self, idx, use_deformed=False):
        normal_file, deformed_file = self.idx_to_file_mapping[idx]
        file_to_sample = deformed_file if use_deformed else normal_file
        base_name = os.path.splitext(os.path.basename(normal_file))[0]
        extent = self.extent_dict[base_name]
        center = self.center_dict[base_name]
        mesh_i = trimesh.load(file_to_sample, process=False)
        sampled_verts, sampled_faces = trimesh.sample.sample_surface(mesh_i, self.nsamples)
        if self.center_and_rescale:
            verts, target_scale, _ = rescale_center(sampled_verts, extent, center)
        else:
            target_scale = torch.Tensor([1])  # No rescaling needed
        sampled_verts = torch.from_numpy(sampled_verts).float()
        sampled_faces = torch.from_numpy(sampled_faces).float()
        return sampled_verts, sampled_faces, target_scale

    def __getitem__(self, idx):
        i, j = self.idx_to_combinations(idx)
        return self.get_pairs(idx)
    
# Classical ICP Helper Functions
def best_fit_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t
# Classical ICP
def classical_icp(src, dst, max_iterations=50, tolerance=1e-6):
    prev_error = float('inf')
    src_aligned = src.copy()
    for i in range(max_iterations):
        tree = KDTree(dst)
        distances, indices = tree.query(src_aligned)
        dst_matched = dst[indices]
        R, t = best_fit_transform(src_aligned, dst_matched)
        src_aligned = (R @ src_aligned.T).T + t
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    final_R, final_t = best_fit_transform(src, src_aligned)
    T = np.eye(4)
    T[:3, :3] = final_R
    T[:3, 3] = final_t
    return src_aligned, T

# Dataloader for Healthy Femur Target Retreival
class ShapePointsetFromTXT:
    def __init__(self, txt_file, nsamples, n_gridpoints, center_and_rescale=True):
        self.txt_file = txt_file
        self.nsamples = nsamples
        self.n_gridpoints = n_gridpoints
        self.center_and_rescale = center_and_rescale

        self.pair_paths = []
        self.normal_path_dict = {}
        self.extent_dict = {}
        self.center_dict = {}

        with open(txt_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                source_path, matched_path = line.strip().split()
                source_path = os.path.normpath(os.path.abspath(source_path))
                matched_path = os.path.normpath(os.path.abspath(matched_path))
                normal_path = self._to_normal_path(matched_path)
                self.pair_paths.append((source_path, matched_path))
                self.normal_path_dict[matched_path] = normal_path

                verts = trimesh.load(normal_path, process=False).vertices
                _, extent, center = rescale_center(verts)
                base_name = os.path.basename(normal_path)
                self.extent_dict[base_name] = extent
                self.center_dict[base_name] = center.numpy()

    def _to_normal_path(self, matched_path):
        base_name = os.path.basename(matched_path)
        patient_id = base_name.split("_")[1]
        normal_file = f"femur_{patient_id}.ply"
        normal_path = matched_path.replace("deformed", "normal")
        normal_path = "/".join(normal_path.split("/")[:-2]) + "/" + normal_file
        return os.path.normpath(normal_path)

    def sample_points(self, mesh_path, nsamples):
        mesh = trimesh.load(mesh_path, process=False)
        verts = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        mesh = Meshes(verts=[verts], faces=[faces])
        sample_points = sample_points_from_meshes(mesh, nsamples)
        return sample_points.squeeze().cpu().numpy()

    def poisson_sampling(self, mesh_path, n_gridpoints, extent, center):
        v, f, n = pcu.load_mesh_vfn(mesh_path)
        v, _, _ = rescale_center(v, extent, center)
        f_i, bc = pcu.sample_mesh_poisson_disk(v, f, n_gridpoints)
        grid = pcu.interpolate_barycentric_coords(f, f_i, bc, v)
        return torch.from_numpy(grid).float()

    def __len__(self):
        return len(self.pair_paths)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        source_path, matched_path = self.pair_paths[idx]
        matched_path = os.path.normpath(os.path.abspath(matched_path))
        normal_path = self.normal_path_dict[matched_path]

        base_name = os.path.basename(normal_path)
        extent = self.extent_dict[base_name]
        center = self.center_dict[base_name]

        verts_i = self.sample_points(source_path, self.nsamples)
        verts_j = self.sample_points(normal_path, self.nsamples)

        if self.center_and_rescale:
            verts_i, _, _ = rescale_center(verts_i, extent, center)
            verts_j, target_scale, _ = rescale_center(verts_j, extent, center)
        else:
            target_scale = torch.tensor([1.0])

        verts_j_np = verts_j if isinstance(verts_j, np.ndarray) else verts_j.detach().cpu().numpy()
        verts_i_np = verts_i if isinstance(verts_i, np.ndarray) else verts_i.detach().cpu().numpy()
        verts_i_icp, _ = classical_icp(verts_i_np, verts_j_np)
        return idx, torch.tensor(verts_i_icp).float(), torch.tensor(verts_j_np).float(), target_scale, normal_path, matched_path


def build(args,
          kwargs,
          logger,
          center_and_rescale=True):
    """
    Create datasets for OsteoDeform
    """
    if not args.sample_surface:
        logger.info("sample vertex")
        fullset = ShapeVertex(
            data_root=args.data_root,
            split="train",
            nsamples=args.samples,
            center_and_rescale=center_and_rescale,
            extent=None,
            center=None
        )
    else:
        logger.info("sample points")
        fullset = ShapePointset(   
            data_root=args.data_root,
            split="train",   
            nsamples=args.samples,
            n_gridpoints= 500,
            center_and_rescale=center_and_rescale,
            extent=None,
            center=None
        )
    train_meshset = ShapeMesh(
        data_root=args.data_root,
        split="train",
        nsamples= args.samples,
        center_and_rescale=center_and_rescale,
        extent=None,
        center=None
    )

    
    if not args.sample_surface:
        valset = ShapeVertex(
            data_root=args.data_root,
            nsamples=args.samples,
            split="val",
            center_and_rescale=center_and_rescale,
            extent=None,
            center=None
        )
    else:
        valset = ShapePointset(
            data_root=args.data_root,
            nsamples=args.samples,
            n_gridpoints= 500,
            split="val",
            center_and_rescale=center_and_rescale,
            extent=None,
            center=None
        )

    val_mesh_set = ShapeMesh(
        data_root=args.data_root,
        split="val",
        nsamples=args.samples,
        center_and_rescale=center_and_rescale,
        extent=None,
        center=None
    )

    if not args.sample_surface:
        testset = ShapeVertex(
            data_root=args.data_root,
            nsamples=args.samples,
            split="test",
            center_and_rescale=center_and_rescale,
            extent=None,
            center=None
        )
    else:
        testset = ShapePointset(
            data_root=args.data_root,
            nsamples=args.samples,
            n_gridpoints= 500,
            split="test",
            center_and_rescale=center_and_rescale,
            extent=None,
            center=None
        )

    test_mesh_set = ShapeMesh(
        data_root=args.data_root,
        split="test",
        nsamples=args.samples,
        center_and_rescale=center_and_rescale,
        extent=None,
        center=None
    )
    shape_analysis_dataset = ShapePointsetFromTXT(
        "output_results_train_val.txt", 5000, 1000, True
    )
    shape_base_instance = ShapeBase(
        data_root=args.data_root,
        split = "train",
        center_and_rescale=center_and_rescale,
        center=None,
        extent=None
    )
    shapepoint_instance = ShapePointset(
        data_root = r"/home/imaging/new_nfd_03_02_2025/split_data_new",
        split = "train",
        nsamples=2000,
        n_gridpoints=1000,
        center_and_rescale=center_and_rescale,
        center=None,
        extent=None
    )
    # Dataloader
    train_points = DataLoader(
        fullset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    train_mesh = DataLoader(
        train_meshset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    val_mesh = DataLoader(
        val_mesh_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    val_points = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )

    test_mesh = DataLoader(
        test_mesh_set,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    test_points = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    shape_analysis = DataLoader(
        shape_analysis_dataset,
        batch_size = 1,
        shuffle=False,
        drop_last=False,
        **kwargs,
        
    )
    return shape_base_instance, shapepoint_instance,  train_mesh, train_points, val_mesh, val_points, test_mesh, test_points, shape_analysis

