import os
import open3d as o3d
import numpy as np
def sample_point_cloud_from_ply(ply_path, num_points=1024):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()
    # Uniformly sample points from the mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points)

all_ply_files= []
all_point_clouds = []
input_directory = r"/home/imaging/new_nfd_03_02_2025/new_dataset_encoder/deformed" # directory name of the 396 osteophyte femur model
                                                                                   # extract only the training and validation deformed files and crreate a new directory 
dirlist = os.listdir(input_directory)
for i in dirlist:
    subpath = os.path.join(input_directory, i)
    subdir = os.listdir(subpath)
    for j in subdir:
        indi_files = os.path.join(subpath, j)
        all_ply_files.append(indi_files)

for i in all_ply_files:
    file_path = i
    points = sample_point_cloud_from_ply(file_path, num_points=5000)

    if points.shape[0] != 5000:
        print(f"Warning: {file_path} had {points.shape[0]} points instead of 1024. Padding or skipping.")
        # Optionally pad or skip
        continue

    all_point_clouds.append(points)

# Convert to numpy array
pc_array = np.stack(all_point_clouds, axis=0)  # Shape: (N, 1024, 3)
print("Shape of pc_array:", pc_array.shape)

# Save to file
np.save("femur_train_val.npy", pc_array) # saving the npy file in the directory (396 osteophytic femur meshes)
# Save file names
with open("train_val_ply_files.txt", "w") as f: # Writes the file path in the txt 
    for path in all_ply_files:
        f.write(f"{path}\n")