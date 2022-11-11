import os
import pandas as pd
from torchvision.io import read_image
import torch
from PIL import Image
import numpy as np


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        #self.image = pd.read_csv(annotations_file)
        self.images = images
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        filename_img = self.imgs[image_id]["file_name"]
        img_path = os.path.join(self.img_dir, filename_img)
        #image = Image.open(img_path).convert("RGB")
        #image = read_image(img_path)
        image = Image.open(img_path)
        image.show()
        image = np.array(image, dtype=np.float32) / 255.0 # normalize image
        #img_shape = image.shape
        label = self.label
        if self.transform:
            image = self.transform(image)
        return image, label
