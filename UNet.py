import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytorch_ssim
import torch.nn.functional as F

print(pytorch_ssim.__version__)

# Define the necessary directories and file paths
coco_dir = 'path_to_coco/train2017'
subset_percentage = 0.02

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
])

# Load the MSCOCO dataset
full_dataset = datasets.ImageFolder(coco_dir, transform=transform)

# Create a subset of 2% of the dataset
subset_size = int(len(full_dataset) * subset_percentage)
indices = np.random.choice(len(full_dataset), subset_size, replace=False)
subset_dataset = Subset(full_dataset, indices)

# DataLoaders
batch_size = 16
train_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

# Load pre-trained VGG16 network
vgg16 = models.vgg16(pretrained=True)
vgg16.features = nn.Sequential(*list(vgg16.features.children())[:16])  # Up to 'block3_conv3'
for param in vgg16.parameters():
    param.requires_grad = False

# Custom loss function for perceptual loss
class PerceptualLoss(nn.Module):
    def __init__(self, vgg):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg

    def forward(self, y_true, y_pred):
        y_true_features = self.vgg(y_true)
        y_pred_features = self.vgg(y_pred)
        loss = nn.functional.mse_loss(y_true_features, y_pred_features)
        return loss

perceptual_loss = PerceptualLoss(vgg16.features)

# Additional metrics (PSNR and SSIM)
def psnr(y_true, y_pred):
    mse = nn.functional.mse_loss(y_true, y_pred)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def ssim(y_true, y_pred):
    return pytorch_ssim.ssim(y_true, y_pred)

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x
    
# Define the image processing network
class ImageProcNet(nn.Module):
    def __init__(self, training_mode=True):
        super(ImageProcNet, self).__init__()
        if training_mode:
            self.model = nn.Sequential(
                nn.MaxPool2d(2),
                GaussianNoise(5),
                nn.Upsample(scale_factor=2, mode='nearest')
            )
        else:
            self.model = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest')
            )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)

image_proc_train = ImageProcNet(training_mode=True)
image_proc_test = ImageProcNet(training_mode=False)

def pixel_shuffle(scale):
    def layer(x):
        return F.pixel_shuffle(x, scale)
    return layer

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual=False):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_residual = use_residual
        if use_residual:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_residual:
            residual = self.residual_conv(residual)
        x += residual
        x = self.relu(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, upscale_factor=2):
        super(UpBlock, self).__init__()
        self.pixel_shuffle = pixel_shuffle(upscale_factor)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.pixel_shuffle(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class RUNet(nn.Module):
    def __init__(self):
        super(RUNet, self).__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down_2 = nn.Sequential(
            nn.MaxPool2d(2),
            DownBlock(64, 64),
            DownBlock(64, 64),
            DownBlock(64, 64),
            DownBlock(64, 128, use_residual=True)
        )
        self.down_3 = nn.Sequential(
            nn.MaxPool2d(2),
            DownBlock(128, 128),
            DownBlock(128, 128),
            DownBlock(128, 128),
            DownBlock(128, 256, use_residual=True)
        )
        self.down_4 = nn.Sequential(
            nn.MaxPool2d(2),
            DownBlock(256, 256),
            DownBlock(256, 256),
            DownBlock(256, 256),
            DownBlock(256, 256),
            DownBlock(256, 256),
            DownBlock(256, 512, use_residual=True)
        )
        self.down_5 = nn.Sequential(
            nn.MaxPool2d(2),
            DownBlock(512, 512),
            DownBlock(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.bottleneck_1 = Bottleneck(512, 1024)
        self.bottleneck_2 = Bottleneck(1024, 512)
        self.up_1 = UpBlock(512, 512, 512, upscale_factor=1)
        self.up_2 = UpBlock(512, 256, 384, upscale_factor=2)
        self.up_3 = UpBlock(384, 128, 256, upscale_factor=2)
        self.up_4 = UpBlock(256, 64, 96, upscale_factor=2)
        self.final_up = pixel_shuffle(2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(96 + 64, 99, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(99, 99, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(99, 3, kernel_size=1, padding=0)
        )

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        bottleneck_1 = self.bottleneck_1(down_5)
        bottleneck_2 = self.bottleneck_2(bottleneck_1)
        up_1 = self.up_1(bottleneck_2, down_5)
        up_2 = self.up_2(up_1, down_4)
        up_3 = self.up_3(up_2, down_3)
        up_4 = self.up_4(up_3, down_2)
        final_up = self.final_up(up_4)
        final_out = self.final_conv(torch.cat((final_up, down_1), dim=1))
        return final_out

model = RUNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = perceptual_loss(y_batch, y_pred)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}')

    # Add validation loop and metric calculations as needed

# Visualization functions (adapted from the original code)
def show_pictures(img_idx, x_batch, y_batch, model):
    fig = plt.figure(figsize=(15, 18))

    ax1 = fig.add_subplot(1, 3, 1)
    with torch.no_grad():
        im = model(x_batch[img_idx].unsqueeze(0))
    im = im.squeeze().permute(1, 2, 0).cpu().numpy()
    ax1.imshow(np.abs(im))
    ax1.set_title('Super Resolution (from LR)')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(x_batch[img_idx].permute(1, 2, 0).cpu().numpy())
    ax2.set_title('Low Resolution')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(y_batch[img_idx].permute(1, 2, 0).cpu().numpy())
    ax3.set_title('Ground truth')

    plt.show()

# Testing the visualization with some training images
for x_batch, y_batch in train_loader:
    show_pictures(0, x_batch, y_batch, model)
    break
