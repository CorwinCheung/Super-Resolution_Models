import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define directories
train_folder = 'path/to/train2017'  # Update this path to your actual train2017 folder

# Custom transform to maintain aspect ratio
class ResizeWithAspectRatio:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return transforms.functional.resize(img, self.size)

# Load dataset
transform_highres = transforms.Compose([
    ResizeWithAspectRatio((800, 800)),
    transforms.ToTensor(),
])

transform_lowres = transforms.Compose([
    ResizeWithAspectRatio((200, 200)),  # downsampling to 1/4th of the size
    ResizeWithAspectRatio((800, 800)),  # upsampling back to original size
    transforms.ToTensor(),
])


# Dataset class to handle both highres and lowres transforms
class SuperResolutionDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.imgs[index]
        image = self.loader(path)
        highres_image = transform_highres(image)
        lowres_image = transform_lowres(image)
        return lowres_image, highres_image

dataset = SuperResolutionDataset(root=train_folder)

# Sample 2% of the dataset
sample_size = int(0.02 * len(dataset))
_, sample_dataset = random_split(dataset, [len(dataset) - sample_size, sample_size])

# Create DataLoader
batch_size = 4
train_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True)

# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, padding=1),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize the autoencoder
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training the autoencoder
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for lowres_imgs, highres_imgs in train_loader:
        lowres_imgs, highres_imgs = lowres_imgs.to(device), highres_imgs.to(device)

        # Forward pass
        outputs = model(lowres_imgs)
        loss = criterion(outputs, highres_imgs)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * lowres_imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# Display some images
def show_images(lowres_images, highres_images, outputs):
    fig, axes = plt.subplots(len(lowres_images), 3, figsize=(20, 8))
    for i in range(len(lowres_images)):
        ax1, ax2, ax3 = axes[i]
        ax1.imshow(lowres_images[i].permute(1, 2, 0).cpu().numpy())
        ax1.set_title("Low Resolution Image")
        ax2.imshow(highres_images[i].permute(1, 2, 0).cpu().numpy())
        ax2.set_title("High Resolution Image")
        ax3.imshow(outputs[i].permute(1, 2, 0).cpu().detach().numpy())
        ax3.set_title("Predicted High Resolution Image")
    plt.show()

model.eval()
with torch.no_grad():
    for lowres_imgs, highres_imgs in train_loader:
        lowres_imgs, highres_imgs = lowres_imgs.to(device), highres_imgs.to(device)
        outputs = model(lowres_imgs)
        show_images(lowres_imgs, highres_imgs, outputs)
        break
