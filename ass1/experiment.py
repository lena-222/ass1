"""! brief Experiment."""

import random
import sys

# from ema_pytorch import EMA
import numpy as np
import torch
import torch.backends
import torch.backends.cudnn
import torch.cuda.amp as amp  # will be used for mixed precision training
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # will be used for visualization loading

from image_dataset import ImageDataset
from pytorch_models import ConvNextTiny
from pytorch_models import ResNet18


def make_reproducible():
    """Set seeds to make the experiment reproducible (see https://pytorch.org/docs/stable/notes/randomness.html)"""
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.Generator().manual_seed(0)


def train(dataloader,
          data,
          model,
          optimizer,
          device,
          criterion,
          cur_iter,
          batch_size,
          use_scaler):
    """Train model one epoch."""

    model.train()

    with tqdm(total=len(data), desc="Training progress:\t\t", bar_format="{l_bar}{bar:50}| \\ [ epoch " +
                                                                         str(cur_iter) + " ]",
              file=sys.stdout) as pbar:
        print("mep1")
        for batch, data in enumerate(dataloader):
            images = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()
            print("mep2")
            print(batch)
            # torch.backends.cudnn.enabled = False
            if use_scaler:
                scaler = amp.GradScaler()
                print("mep3")
                # TODO add Mixed Precision Training
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    print("mep4")
                    y_hat = model(images)
                    loss = criterion(y_hat, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                y_hat = model(images)
                loss = criterion(y_hat, labels)
                loss.backward()
                optimizer.step()

        pbar.update(batch_size)
        # torch.cuda.synchronize()


def val(dataloader,
        data,
        model,
        device,
        cur_iter,
        num_classes,
        batch_size):
    """Evaluate model on validation data."""
    # model.train()
    # val_loss_list = []

    all_labels = torch.zeros(model.out_features)
    correct_labels = torch.zeros(model.out_features)

    model.eval()

    # scaler = amp.GradScaler()

    with torch.no_grad():

        with tqdm(total=len(data), desc="Validation progress:\t", bar_format="{l_bar}{bar:50}| \\ [ epoch " +
                                                                             str(cur_iter) + " ]",
                  file=sys.stdout) as pbar:

            for batch, data in enumerate(dataloader):
                images = data[0].to(device)
                labels = data[1].to(device)
                output = model(images)

                # use mean per class accuracy as evaluation metric
                for label, prediction in zip(labels, output):
                    pred_label = torch.argmax(prediction)

                    if pred_label == label:
                        correct_labels[label] += 1

                    all_labels[label] += 1

            pbar.update(batch_size)
            # torch.cuda.synchronize()

    # mean per class accuracy
    not_zero = []
    for i in range(0, num_classes):
        if all_labels[i] > 0:
            not_zero.append(i)

    numerator = [correct_labels[i] / all_labels[i] for i in not_zero]
    accuracy = sum(numerator) / len(not_zero)

    print("Mean per class accuracy: ", accuracy.item())

    return accuracy.item()


def evaluate_model(accuracy_per_epoch):
    print("Writing output")
    writer = SummaryWriter()

    for epoch in range(0, len(accuracy_per_epoch)):
        writer.add_scalar('mean per class accuracy - validation', accuracy_per_epoch[epoch], epoch)

    writer.close()


def model_train_and_eval(dataset_path,
                         transform_type,
                         model_name,
                         batch_size,
                         num_classes,
                         epochs,
                         learning_rate,
                         use_scheduler=False,
                         ema=False,
                         ema_rate=None):
    print("Experiment will be started!")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Used device: ", device)

    # make experiment reproducible
    print("Seeds are set!")
    make_reproducible()

    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True

    # create dataset with custom Dataset class ImageDataset
    print("Start loading ImageDataSet...")
    full_dataset = ImageDataset(data_path=dataset_path, transform_type=transform_type)

    # Split data into train and validation data
    print("Start loading ImageDataSet...")
    train_size = int(0.6 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    # train_data.dataset = copy.copy(full_dataset)  # is needed to work - still only train data is used

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1)

    # set model
    if model_name == 'ResNet18':
        model = ResNet18().to(device)
    elif model_name == 'ConvNext':
        model = ConvNextTiny().to(device)
    else:
        raise ValueError("Model not found!")
    print("Used Model: ", model_name)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    # criterion for classification
    criterion = nn.CrossEntropyLoss()
    accuracy_per_epoch = np.zeros(epochs)

    # set adam optimizer and a learning rate of 0.0001
    # eventually create schedular and use sdg optimizer,
    # since there is no learning scheduler for adam optimizer
    if use_scheduler:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    else:
        optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)

    num_batches = np.ceil(len(train_data) / batch_size)
    best_acc = 0.0  # save the best accuracy later
    use_scaler = True

    print("Start training...")

    for e in range(epochs):
        print("Current epoch: ", e)
        cur_iter = e * num_batches
        train(dataloader=train_dataloader, data=train_data,
              model=model, optimizer=optimizer, device=device, criterion=criterion,
              cur_iter=cur_iter, batch_size=batch_size, use_scaler=use_scaler)
        cur_iter += num_batches
        # val_acc, _, _ = get_test_accuracy(model, val_dataloader, device)

        val_acc = val(dataloader=val_dataloader, data=val_data, model=model,
                      device=device, cur_iter=cur_iter, num_classes=num_classes, batch_size=batch_size).item()
        accuracy_per_epoch[e] = val_acc

        if val_acc >= max(accuracy_per_epoch):
            # best_acc = val_acc
            torch.save(model.state_dict(), "output/best_model_weights.pth")
            torch.save(optimizer.state_dict(), "output/best_optimizer_state.pth")
            print("Saved best PyTorch Model State to output/model_weights.pth")
        if use_scheduler:
            scheduler.step()

        torch.save(model.state_dict(), "output/model_weights_ConvNeXt_Tiny.pth")
        print("Saved PyTorch Model State to output/model_weightsConvNeXt_Tiny.pth")
        torch.save(optimizer.state_dict(), "output/optimizer_stateConvNeXt_Tiny.pth")
        print("Saved PyTorch Optimizer State to output/optimizer_weightsConvNeXt_Tiny.pth")
        evaluate_model(accuracy_per_epoch=accuracy_per_epoch)
        print("Evaluation succeeded")
        model.load_state_dict(torch.load('output/model_weightsConvNeXt_Tiny.pth'))
        optimizer.load_state_dict(torch.load('output/optimizer_stateConvNeXt_Tiny.pth'))

    print("Done!")


# TODO Initialize second model with EMA-rates and the same weights from before
''' NOT USED YET 
https://github.com/lucidrains/ema-pytorch

# wrap your neural network, specify the decay (beta)

ema = EMA(
    net,
    beta = 0.998,              # exponential moving average factor
    update_after_step = 100,    # only after this number of .update() calls will it start updating
    update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
)

# mutate your network, with SGD or otherwise

with torch.no_grad():
    net.weight.copy_(torch.randn_like(net.weight))
    net.bias.copy_(torch.randn_like(net.bias))

# you will call the update function on your moving average wrapper

ema.update()

# then, later on, you can invoke the EMA model the same way as your network

data = torch.randn(1, 512)

output     = net(data)
ema_output = ema(data)

# if you want to save your ema model, it is recommended you save the entire wrapper
# as it contains the number of steps taken (there is a warmup logic in there, recommended by @crowsonkb, validated for a number of projects now)
# however, if you wish to access the copy of your model with EMA, then it will live at ema.ema_model
'''
