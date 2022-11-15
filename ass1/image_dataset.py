import os
import random
from glob import glob

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image


# imagenet_stats: ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
def transform():
    """Preprocess data with imagenet stats for the first tasks and for validation data"""
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               transforms.Resize(size=[128, 128])])


# add data augmentation: random cropping; horizontal flipping
def transform_geometric():
    """Preprocess data with geometric augmentation"""
    sample = random.randrange(100, 128)
    return transforms.Compose([transforms.ToTensor(),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomCrop(size=sample, pad_if_needed=True),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               transforms.Resize(size=[128, 128])])


# add ColorJitter augmentation
def transform_colorjitter():
    """Preprocess data with geometric and color-jitter augmentation"""
    sample = random.randrange(60, 128)
    return transforms.Compose([transforms.ToTensor(),
                               transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomCrop(size=sample, pad_if_needed=True),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               transforms.Resize(size=[128, 128])])

def data_preloading(data_path):
    """load image_dirs and label_ids from data_path"""
    subdir = sorted(glob(os.path.join(data_path, "*")))
    idx = 0
    images = []
    labels = []
    for sub in subdir:
        for image_dir in sorted(glob(os.path.join(sub, "*.jpg"))):
            # print(image_dir)
            images.append(image_dir)
            labels.append(idx)
        idx += 1
    return images, labels


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, transform_type):
        """load image_dirs and label_ids from data_path"""
        # self.image = pd.read_csv(annotations_file)
        # self.images = torch(images)
        # self.labels = torch(labels)
        # self.images = images
        # self.labels = labels
        self.images, self.labels = data_preloading(data_path)
        if transform_type == "geometric":
            self.transform = transform_geometric()
        elif transform_type == "colorjitter":
            self.transform = transform_colorjitter()
        else:
            self.transform = transform()
        self.num_examples = len(self.images)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        # print("inside ImageDataset __getitem__")
        # print(idx)
        filename_img = self.images[idx]
        # print(filename_img)
        # img_path = os.path.join(self.images, filename_img)
        image = Image.open(filename_img).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # image = np.array(image, dtype=np.float32) / 255.0 # normalize image
        # image.show()
        label = np.asarray(self.labels[idx])
        label = torch.from_numpy(label.copy()).long()

        return image, label
