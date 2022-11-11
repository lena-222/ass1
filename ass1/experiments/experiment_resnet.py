"""! @brief Experiment."""

# import signal
import torch
import copy
from sacred import Experiment
# from util.general_utils import signal_handler
from models.pytorch_models import ResNet18
from models.ImageDataset import ImageDataset

import numpy as np
import random

# make experiment reproducible with notations from:
# https://pytorch.org/docs/stable/notes/randomness.html
def make_reproducible():
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

ex = Experiment()

def train(dataloader, model, optimizer, device, cur_iter, scaler=None):
    """Train model one epoch."""

    model.train()
    for batch, (images, targets) in enumerate(dataloader):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            # TODO use mean per class accuracy as evaluation metric
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


def val(dataloader, model, device, cur_iter, bol_log=True):
    """Evaluate model on validation data."""
    model.train()
    val_loss_list = []

    with torch.no_grad():
        for batch, (images, targets) in enumerate(dataloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            # TODO use mean per class accuracy as evaluation metric
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
    #signal.signal(signal.SIGINT, signal_handler)
    #cfg = _config
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Implement machine learning things here.
    #print(cfg)
    
#    train_data  = CocoImageDataset("data/initial_data/train/small_big_training.json", "data/initial_data/train/", transform=get_transform(True))    
#    test_data  = CocoImageDataset("data/initial_data/test/labels_middle_2022-09-01-04-37-47.json", "data/initial_data/test/", transform=get_transform(False))
    
    full_dataset  = ImageDataset("data/initial_data/test/labels_middle_2022-09-01-04-37-47.json", "data/initial_data/test/", transform=get_transform(False))
    train_size = int(0.6 * len(full_dataset))
#    test_size = len(full_dataset) - train_size
    val_size = len(full_dataset) - train_size

    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_data.dataset = copy.copy(full_dataset) # is needed to work - still only train data is used
    #train_data.dataset.transform = get_transform(True)
    #val_data.dataset.transform = get_transform(False)

    batch_size = 64

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1)

    #num_classes =

    model = ResNet18().model.to(device)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    # use of adam optimizer and an learning rate of 0.0001
    optimizer = torch.optim.Adam(trainable_parameters, lr=0.0001)
    # set 30 epochs of training with a batchsize of 64
    epochs = 30
    num_batches = np.ceil(len(train_data) / batch_size)
    best_acc = 0
    for e in range(epochs):
        cur_iter = e * num_batches
        train(train_dataloader, model, optimizer, device, cur_iter, scaler=None)
        cur_iter += num_batches
        #val_acc, _, _ = get_test_accuracy(model, val_dataloader, device)
        val(val_dataloader, model, device, cur_iter)

#        if val_acc >= best_acc:
#            best_acc = val_acc

    torch.save(model.state_dict(), "output/model.pth")
    print("Saved PyTorch Model State to output/model.pth")

    print("Done!")
    
    