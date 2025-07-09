import numpy as np
import time
import AEutils
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
from Dataloaders import GetDataLoaders

# -------- Model Definition --------
class PointCloudAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAE, self).__init__()
        self.latent_size = latent_size
        self.point_size = point_size

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, latent_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(latent_size)

        self.dec1 = nn.Linear(latent_size, 512)
        self.dec2 = nn.Linear(512, 1024)
        self.dec3 = nn.Linear(1024, point_size * 3)

    def encoder(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        return x.view(-1, self.latent_size)

    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

# -------- Utility Functions --------
def normalize_point_cloud(pc):
    pc = pc - np.mean(pc, axis=0)
    scale = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    return pc / scale

def test_batch(data, net, device):
    with torch.no_grad():
        data = data.to(device)
        output = net(data.permute(0, 2, 1))
        loss, _ = chamfer_distance(data, output)
    return loss.item(), output.cpu()

def train_epoch(train_loader, net, optimizer, device):
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=True)
    for i, data in enumerate(progress_bar):
        optimizer.zero_grad()
        data = data.to(device)
        output = net(data.permute(0, 2, 1))
        loss, _ = chamfer_distance(data, output)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return epoch_loss / (i + 1)

def test_epoch(test_loader, net, device):
    with torch.no_grad():
        epoch_loss = 0
        progress_bar = tqdm(test_loader, desc="Testing", leave=True)
        for i, data in enumerate(progress_bar):
            loss, _ = test_batch(data, net, device)
            epoch_loss += loss
            progress_bar.set_postfix(loss=loss)
    return epoch_loss / (i + 1)

def save_checkpoint(model, optimizer, epoch, train_loss_list, test_loss_list, output_folder):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_list': train_loss_list,
        'test_loss_list': test_loss_list,
    }
    torch.save(checkpoint, f"{output_folder}/checkpoint_epoch_{epoch}.pt")

# -------- Training Code (Main) --------
if __name__ == "__main__":
    # SETTINGS
    batch_size = 8
    output_folder = "output1/" # Output directory 
    save_results = True
    use_GPU = True
    latent_size = 128

    # Load and normalize dataset
    pc_array = np.load("/home/imaging/new_nfd_03_02_2025/femur_train_val.npy") # path of the .npy file
    pc_array = np.array([normalize_point_cloud(pc) for pc in pc_array])
    print("Loaded and normalized dataset with shape:", pc_array.shape)

    train_loader, test_loader = GetDataLoaders(npArray=pc_array, batch_size=batch_size)
    point_size = len(train_loader.dataset[0])
    print("Point size:", point_size)

    net = PointCloudAE(point_size, latent_size)

    device = torch.device("cuda:0" if use_GPU and torch.cuda.is_available() else "cpu")
    if use_GPU and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.000001)

    if save_results:
        AEutils.clear_folder(output_folder)

    train_loss_list = []
    test_loss_list = []

    for i in tqdm(range(1501), desc="Epochs"):
        startTime = time.time()

        train_loss = train_epoch(train_loader, net, optimizer, device)
        test_loss = test_epoch(test_loader, net, device)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        epoch_time = time.time() - startTime
        log_str = f"Epoch {i} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Time: {epoch_time:.2f}s\n"

        if save_results:
            with open(output_folder + "prints.txt", "a") as f:
                f.write(log_str)

            plt.plot(train_loss_list, label="Train")
            plt.plot(test_loss_list, label="Test")
            plt.legend()
            plt.savefig(output_folder + "loss.png")
            plt.close()

            if i % 50 == 0:
                test_samples = next(iter(test_loader))
                loss, test_output = test_batch(test_samples, net, device)
                AEutils.plotPCbatch(test_samples, test_output, show=False, save=True, name=output_folder + f"epoch_{i}")
                save_checkpoint(net, optimizer, i, train_loss_list, test_loss_list, output_folder)
        else:
            print(log_str)
            plt.plot(train_loss_list, label="Train")
            plt.plot(test_loss_list, label="Test")
            plt.legend()
            plt.show()
