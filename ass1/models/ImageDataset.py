import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import random


# imagenet_stats: ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
def transform():
    return transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                transforms.Resize(size=[128, 128])])

# add data augmentation: random cropping; horizontal flipping
def transform_geometric():
    return transforms.Compose([transforms.ToTensor(),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomCrop(random.sample(range(60, 128)), random.sample(range(60, 128))),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                          transforms.Resize(size=[128, 128])])

# add ColorJitter augmentation
def transform_colorjitter():
    return transforms.Compose([transforms.ToTensor(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
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

        image_id = self.ids[idx]
        filename_img = self.imgs[image_id]["file_name"]
        img_path = os.path.join(self.img_dir, filename_img)
        image = Image.open(img_path).convert("RGB")
        #image = read_image(img_path)
        #image = Image.open(img_path)
        image.show()

        #image = np.array(image, dtype=np.float32) / 255.0 # normalize image
        label = np.asarray(self.labels[idx])
        label = torch.from_numpy(label.copy()).long()
        # normalize and fix size at 128x128
        if self.transform:
            image = self.transform(image)
        # add geometric augmentation
        if self.transform_geometric:
            image = self.transform_geometric(image)
        # add ColorJitter augmentation
        if self.transform_with_colorjitter:
            image = self.transform_with_colorjitter(image)
        return image, label
