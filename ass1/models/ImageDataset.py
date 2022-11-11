import os
import pandas as pd
from torchvision.io import read_image
import torch
from PIL import Image
import numpy as np
import albumentations
import albumentations.pytorch
import torchvision.transforms as transforms

def get_transform(train):
    """Data augmentation pipeline."""
    if train:
        return albumentations.Compose(
            [
                albumentations.Flip(0.5),
                albumentations.MotionBlur(p=0.2),
                #albumentations.RandomResizedCrop(900, 900),
                albumentations.augmentations.geometric.resize.Resize(900, 900),
                albumentations.pytorch.ToTensorV2(transpose_mask=True),
            ],
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )
    # define the validation transforms
    else:
        return albumentations.Compose(
            [albumentations.augmentations.geometric.resize.Resize(900, 900),albumentations.pytorch.ToTensorV2(transpose_mask=True),],
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )


# TODO add data augmentation: random cropping; horizontal flipping
# imagenet_stats: ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# TODO Size 128x128;
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# TODO add ColorJitter augmentation
transform_with_colorjitter = transforms.Compose([transforms.ToTensor(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform, transform_with_colorjitter):
        #self.image = pd.read_csv(annotations_file)
        #self.images = torch(images)
        #self.labels = torch(labels)
        self.images = images
        self.labels = labels
        self.transform = transform
        self.transform = transform_with_colorjitter
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

        image = np.array(image, dtype=np.float32) / 255.0 # normalize image
        #img_shape = image.shape
        label = self.label
        if self.transform:
            image = self.transform(image)
        if self.transform_with_colorjitter:
            image = self.transform_with_colorjitter(image)
        return image, label
