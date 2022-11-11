"""! @brief Experiment."""

import signal
import torch
import copy
from sacred import Experiment
from util.general_utils import signal_handler
import albumentations
import albumentations.pytorch
from models.pytorch_models import MaskRCNNResnet
from models.coco_image_dataset import CocoImageDataset

import numpy as np

ex = Experiment()

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


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


def train(dataloader, model, optimizer, device, cur_iter, scaler=None):
    """Train model one epoch."""

    model.train()
    for batch, (images, targets) in enumerate(dataloader):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        new_iter = cur_iter + batch
        ex.log_scalar("training.loss", losses.item(), new_iter)


def test(dataloader, model, device, cur_iter, bol_log=True):
    """Evaluate model on validation data."""
    model.train()
    val_loss_list = []

    with torch.no_grad():
        for batch, (images, targets) in enumerate(dataloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            val_loss_list.append(loss_value)

    val_loss_mean = np.mean(val_loss_list)
    print("val loss: ", val_loss_mean)
    if bol_log:
        ex.log_scalar("validation.loss", val_loss_mean, cur_iter)

    return val_loss_mean


@ex.main
def run(_config):
    """Register signal handler."""
    signal.signal(signal.SIGINT, signal_handler)
    cfg = _config
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Implement machine learning things here.
    print(cfg)
    
#    train_data  = CocoImageDataset("data/initial_data/train/small_big_training.json", "data/initial_data/train/", transform=get_transform(True))    
#    test_data  = CocoImageDataset("data/initial_data/test/labels_middle_2022-09-01-04-37-47.json", "data/initial_data/test/", transform=get_transform(False))
    
    full_dataset  = CocoImageDataset("data/initial_data/test/labels_middle_2022-09-01-04-37-47.json", "data/initial_data/test/", transform=get_transform(False))
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    
#    train_data  = CocoImageDataset("data/bottle/bottle_coco_training.json", "", transform=get_transform(True))    
#    test_data  = CocoImageDataset("data/bottle/bottle_coco_testing.json", "", transform=get_transform(False))    
    
    batch_size = 2

    train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_data.dataset = copy.copy(full_dataset) # is needed to work - still only train data is used
    train_data.dataset.transform = get_transform(True)
    test_data.dataset.transform = get_transform(False)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, collate_fn=collate_fn)
    
    model = MaskRCNNResnet(2).model.to(device)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(trainable_parameters, lr=0.0002)

    epochs = 100
    num_batches = np.ceil(len(train_data) / batch_size)
    best_acc = 0
    for e in range(epochs):
        cur_iter = e * num_batches
        train(train_dataloader, model, optimizer, device, cur_iter, scaler=None)
        cur_iter += num_batches
        #val_acc, _, _ = get_test_accuracy(model, val_dataloader, device)
        test(test_dataloader, model, device, cur_iter)

#        if val_acc >= best_acc:
#            best_acc = val_acc

    torch.save(model.state_dict(), "output/model.pth")
    print("Saved PyTorch Model State to output/model.pth")

    print("Done!")
    
    