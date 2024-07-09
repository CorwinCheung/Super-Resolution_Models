import os
import random
import numpy as np
import torch
import torch.utils.data as data
import skimage.io as io
from skimage import transform
from skimage.util import img_as_float
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
from matplotlib import pyplot as plt
import tqdm

# Constants
data_dir = 'images2017'
output_dir = 'output_images'
model_save_path = 'checkpoints'
border = 30
overlap = 30
patch_size = 96
num_patches = 20
train_split = 0.02

# Create necessary directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)

# List images
all_images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
sampled_images = random.sample(all_images, int(len(all_images) * train_split))

# Helper functions
def add_border(img, n_pixels):
    h, w, c = img.shape
    img_with_border = np.empty((h + 2*n_pixels, w + 2*n_pixels, c), np.float32)
    img_with_border[n_pixels:n_pixels+h, n_pixels:n_pixels+w, :] = img
    img_with_border[n_pixels:n_pixels+h, 0:n_pixels, :] = img[:, :n_pixels, :][:, ::-1, :]
    img_with_border[n_pixels:n_pixels+h, -n_pixels:, :] = img[:, -n_pixels:, :][:, ::-1, :]
    img_with_border[:n_pixels, :, :] = img_with_border[n_pixels:2*n_pixels, :, :][::-1, :, :]
    img_with_border[-n_pixels:, :, :] = img_with_border[-2*n_pixels:-n_pixels, :, :][::-1, :, :]
    return img_with_border

def remove_border(img, n_pixels):
    return img[n_pixels:-n_pixels, n_pixels:-n_pixels, :]

def preprocess_image(img):
    img = img_as_float(img)
    img = img.transpose(2, 0, 1)
    return img

def restore_image(tensor):
    x = tensor.transpose(1, 2, 0)
    return x

def upscale_image(img_np, model):
    img = add_border(img_np, border)
    img = preprocess_image(img)
    img = np.expand_dims(img, 0).astype(np.float32)
    x = torch.from_numpy(img).cuda()
    with torch.no_grad():
        y = model(x).data[0].cpu().numpy()
    y = restore_image(y)
    y = np.clip(y, 0, 1)
    y = remove_border(y, 2*border)
    return y

def split_image(img, overlap):
    h, w, c = img.shape
    x1, y1 = w // 2, h // 2
    return (
        img[:y1+overlap, :x1+overlap, :],
        img[:y1+overlap, x1-overlap:, :],
        img[y1-overlap:, :x1+overlap, :],
        img[y1-overlap:, x1-overlap:, :]
    )

def remove_overlap(split_imgs, overlap):
    return [
        split_imgs[0][:-overlap, :-overlap, :],
        split_imgs[1][:-overlap, overlap:, :],
        split_imgs[2][overlap:, :-overlap, :],
        split_imgs[3][overlap:, overlap:, :]
    ]

def check_and_upscale(img, model):
    h, w, c = img.shape
    if h > 600 or w > 600:
        split_imgs = split_image(img, overlap)
        split_res = [check_and_upscale(x, model) for x in split_imgs]
        split_res = remove_overlap(split_res, 2*overlap)
        res_image = np.empty((2*h, 2*w, c), dtype=np.float32)
        y1, x1, _ = split_res[0].shape
        res_image[:y1, :x1, :] = split_res[0]
        res_image[:y1, x1:2*w, :] = split_res[1]
        res_image[y1:2*h, :x1, :] = split_res[2]
        res_image[y1:2*h, x1:2*w, :] = split_res[3]
        return res_image
    else:
        return upscale_image(img, model)

# Dataset
class MSCOCODataset(data.Dataset):
    def __init__(self, file_list, hr_patch_size=128, patches_per_img=50):
        self.file_list = file_list
        self.hr_patch_size = hr_patch_size
        self.patches_per_img = patches_per_img
        self.cur_image = None
        self.cur_file_idx = 0
        self.cur_patch = 0
        self.load_next_image()

    def load_next_image(self):
        self.cur_image = imread(self.file_list[self.cur_file_idx])
        self.cur_file_idx = (self.cur_file_idx + 1) % len(self.file_list)

    def __getitem__(self, idx):
        if self.cur_patch == self.patches_per_img:
            self.cur_patch = 0
            self.load_next_image()
        patch = self.get_patch()
        patch = self.augment(patch)
        hr, lr = self.get_tensor_pair(patch)
        self.cur_patch += 1
        return torch.tensor(hr, dtype=torch.float32), torch.tensor(lr, dtype=torch.float32)

    def __len__(self):
        return len(self.file_list) * self.patches_per_img

    def get_patch(self):
        h, w, _ = self.cur_image.shape
        y = random.randint(0, h - self.hr_patch_size)
        x = random.randint(0, w - self.hr_patch_size)
        return self.cur_image[y:y+self.hr_patch_size, x:x+self.hr_patch_size, :]

    def augment(self, patch):
        if random.choice([True, False]):
            patch = np.fliplr(patch)
        if random.choice([True, False]):
            patch = np.flipud(patch)
        angle = random.choice([0, 90, 180, 270])
        return transform.rotate(patch, angle)

    def get_tensor_pair(self, patch):
        hr_patch = preprocess_image(patch)
        lr_patch = transform.resize(patch, (self.hr_patch_size//2, self.hr_patch_size//2, 3), mode='reflect')
        lr_patch = preprocess_image(lr_patch)
        return hr_patch, lr_patch

# Model
def create_conv(n_feat, kernel_size, bias=True):
    return nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias)

class ResBlock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        modules_body = [create_conv(n_feat, kernel_size, bias), act]
        if bn: modules_body.append(nn.BatchNorm2d(n_feat))
        modules_body.append(create_conv(n_feat, kernel_size, bias))
        if bn: modules_body.append(nn.BatchNorm2d(n_feat))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Upscale2x(nn.Sequential):
    def __init__(self, n_feat, bias=True):
        modules = [nn.Conv2d(n_feat, n_feat * 4, 3, padding=1, bias=bias), nn.PixelShuffle(2)]
        super(Upscale2x, self).__init__(*modules)

class ResNet(nn.Module):
    def __init__(self, n_resblocks, n_features):
        super(ResNet, self).__init__()
        self.start = nn.Conv2d(3, n_features, 3, padding=1, bias=True)
        resblocks = [ResBlock(n_features, 3, res_scale=0.15) for _ in range(n_resblocks)]
        resblocks.append(nn.Conv2d(n_features, n_features, 3, padding=1, bias=True))
        self.resblocks = nn.Sequential(*resblocks)
        self.upsampler = Upscale2x(n_features)
        self.end = nn.Conv2d(n_features, 3, 3, padding=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        xs = self.start(x)
        x = self.resblocks(xs)
        x = self.upsampler(x + 0.15*xs)
        x = self.end(x)
        return x

# Training function
def train_epoch(optimizer, net, train_loader, loss_fn):
    net.train()
    for hr, lr in tqdm(train_loader, desc="Training"):
        hr, lr = hr.cuda(), lr.cuda()
        res = net(lr)
        loss = loss_fn(res, hr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Main training loop
def main():
    train_dataset = MSCOCODataset(sampled_images, patch_size, num_patches)
    train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    net = ResNet(32, 128).cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()

    for epoch in range(15):
        print(f"Epoch {epoch+1}")
        train_epoch(optimizer, net, train_loader, loss_fn)
        torch.save(net.state_dict(), os.path.join(model_save_path, f'epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    main()
