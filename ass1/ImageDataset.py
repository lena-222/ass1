import os
from glob import glob

import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random


# imagenet_stats: ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
def transform():
    return transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                transforms.Resize(size=[128, 128])])

# add data augmentation: random cropping; horizontal flipping
def transform_geometric():
    sample = random.randrange(60, 128)
    return transforms.Compose([transforms.ToTensor(),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomCrop(sample,sample),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                          transforms.Resize(size=[128, 128])])

# add ColorJitter augmentation
def transform_colorjitter():
    sample = random.randrange(60, 128)
    return transforms.Compose([transforms.ToTensor(),
                               transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomCrop(sample, sample),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               transforms.Resize(size=[128, 128])])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform_type):
        #self.image = pd.read_csv(annotations_file)
        #self.images = torch(images)
        #self.labels = torch(labels)
        self.images = images
        self.labels = labels
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
        #print("inside ImageDataset __getitem__")
        #print(idx)
        filename_img = self.images[idx]
        #print(filename_img)
        #img_path = os.path.join(self.images, filename_img)
        image = Image.open(filename_img).convert("RGB")
        if self.transform:
            image = self.transform(image)
        #image = np.array(image, dtype=np.float32) / 255.0 # normalize image
        #image.show()
        label = np.asarray(self.labels[idx])
        label = torch.from_numpy(label.copy()).long()

        return image, label
