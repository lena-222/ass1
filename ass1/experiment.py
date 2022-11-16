"""! brief Experiment."""

import random
from datetime import datetime

# from ema_pytorch import EMA
import numpy as np
import torch
import torch.backends
import torch.backends.cudnn
import torch.cuda.amp as amp  # will be used for mixed precision training
import torch.nn as nn

from tqdm import tqdm  # will be used for visualization loading

import torch.utils.data
from torch.utils.data import Subset
from image_dataset import ImageDataset

# imports from my project
from pytorch_models import ConvNextTiny
from pytorch_models import ResNet18

from utils import save_plots, save_model, load_model, \
    make_reproducible  # , evaluate_model#, load_best_accuracy, save_best_accuracy



def train(dataloader,
          model,
          optimizer,
          device,
          criterion,
          cur_iter,
          use_scaler,
          num_classes):
    """Train model one epoch."""

    #torch.backends.cudnn.benchmark = True
    print('Training')
    model.train()

    all_labels = torch.zeros(model.out_features)
    correct_labels = torch.zeros(model.out_features)

    #train_running_correct = 0.0
    counter = 0.0
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        counter += 1
        '''
        if counter == 3 :
            print("Training of epoch " + str(cur_iter) + " completed.")
            accuracy = train_running_correct / len(dataloader.dataset)
            return accuracy
        '''
        images = data[0].to(device)
        labels = data[1].to(device)

        optimizer.zero_grad()
        # torch.backends.cudnn.enabled = False
        if use_scaler:
            scaler = amp.GradScaler()
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, labels)
                # calculate the accuracy
                #_, preds = torch.max(output.data, 1)
                #train_running_correct += (preds == labels).sum().item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        # use mean per class accuracy as evaluation metric
        for label, prediction in zip(labels, output):
            pred_label = torch.argmax(prediction)

            if pred_label == label:
                correct_labels[label] += 1

            all_labels[label] += 1

    # torch.cuda.synchronize()
    # accuracy = train_running_correct / len(dataloader.dataset)
    # mean per class accuracy
    not_zero = []
    for i in range(0, num_classes):
        if all_labels[i] > 0:
            not_zero.append(i)
    numerator = [correct_labels[i] / all_labels[i] for i in not_zero]
    accuracy = sum(numerator) / len(not_zero)
    print("Training of epoch " + str(cur_iter) + " completed.")

    return accuracy


def val(dataloader,
        model,
        device,
        cur_iter,
        num_classes):

    """Evaluate model on validation data."""
    # model.train()
    # val_loss_list = []
    print('Validation')

    all_labels = torch.zeros(model.out_features)
    correct_labels = torch.zeros(model.out_features)

    #torch.backends.cudnn.benchmark = True
    model.eval()

    #valid_running_correct = torch.empty(num_classes)
    #torch.zeros_like(valid_running_correct)
    counter = 0

    with torch.no_grad():

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                counter += 1
            #for batch, data in enumerate(dataloader):
                '''
                if counter == 3:
                    print("Validation of epoch " + str(cur_iter) + " completed.")
                    not_zero = []
                    for i in range(0, num_classes):
                        if all_labels[i] > 0:
                            not_zero.append(i)

                    numerator = [correct_labels[i] / all_labels[i] for i in not_zero]
                    accuracy = sum(numerator) / len(not_zero)
                    #accuracy = train_running_correct / len(dataloader.dataset)
                    return accuracy
                '''
                images = data[0].to(device)
                labels = data[1].to(device)
                # forward pass
                output = model(images)

                # use mean per class accuracy as evaluation metric
                for label, prediction in zip(labels, output):
                    pred_label = torch.argmax(prediction)

                    if pred_label == label:
                        correct_labels[label] += 1

                    all_labels[label] += 1

                #label, preds = torch.max(output.data, 1)
                #valid_running_correct[label] += (preds == labels[label]).sum().item()

            # torch.cuda.synchronize()

    # mean per class accuracy
    not_zero = []
    for i in range(0, num_classes):
        if all_labels[i] > 0:
            not_zero.append(i)

    numerator = [correct_labels[i] / all_labels[i] for i in not_zero]
    accuracy = sum(numerator) / len(not_zero)
    #valid_running_correct
    #epoch_acc = valid_running_correct / len(dataloader.dataset)

    print("Validation of epoch " + str(cur_iter) + " completed.")
    #print("Mean per class accuracy: ", epoch_acc)

    return accuracy


def model_train_and_eval(best_model_output_path,
                         plot_path,
                         output_path,
                         dataset_path,
                         transform_type,
                         model_name,
                         batch_size,
                         num_classes,
                         epochs,
                         learning_rate,
                         num_workers=16,
                         use_scheduler=False,
                         ema=False,
                         ema_rate=None):

    print("Experiment will be started!")

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S") # dd/mm/YY H:M:S
    print("start date and time = ", dt_string)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Used device: ", device)

    # make experiment reproducible
    print("Seeds are set!")
    worker_seed = make_reproducible()

    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True

    # create dataset with custom Dataset class ImageDataset
    print("Start loading ImageDataSet...")
    train_dataset = ImageDataset(data_path=dataset_path, transform_type=transform_type)
    val_dataset = ImageDataset(data_path=dataset_path, transform_type="transform")

    # Split data into train and validation data
    print("Split dataset into train and validation set...")

    # get the training dataset size, need this to calculate the...
    # number if validation images
    # validation size = number of validation images
    valid_size = int(0.4 * len(train_dataset))
    # all the indices from the training set
    indices = torch.randperm(len(train_dataset)).tolist()
    train_data = Subset(train_dataset, indices[:-valid_size])
    val_data = Subset(val_dataset, indices[-valid_size:])
    print(f"Total training images: {len(train_data)}")
    print(f"Total validation images: {len(val_data)}")

    torch.Generator().manual_seed(0)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers,
                                                   shuffle=True, generator=torch.Generator())
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True, drop_last=True,
                                                 worker_init_fn=worker_seed, num_workers=num_workers,
                                                 generator=torch.Generator())

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

    # set adam optimizer and a learning rate of 0.0001
    # eventually create schedular and use sdg optimizer,
    # since there is no learning scheduler for adam optimizer
    if use_scheduler:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    else:
        optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
        scheduler = None

    train_acc = []
    val_acc = []
    best_acc_run = 0
    best_accuracy = 0

    print("Start training...")

    start_e = 1
    for e in range(start_e, epochs + 1):

        try:
            e, model, optimizer, criterion, best_acc_run = load_model(e=e, model=model, optimizer=optimizer,
                                                     criterion=criterion,accuracy=best_acc_run,
                                                     output_path=output_path)
            _, _, _, _, best_accuracy = load_model(e=None, model=None, optimizer=None, criterion=criterion,
                                                   accuracy=best_accuracy, output_path=best_model_output_path)
        except FileNotFoundError:
            pass

        print("Current epoch: ", e)
        #cur_iter = e * num_batches
        train_acc_e = train(dataloader=train_dataloader, model=model, optimizer=optimizer, device=device, criterion=criterion,
                            cur_iter=e, use_scaler=True, num_classes=num_classes)
        #cur_iter += num_batches
        # val_acc, _, _ = get_test_accuracy(model, val_dataloader, device)

        val_acc_e = val(dataloader=val_dataloader, model=model,
                      device=device, cur_iter=e, num_classes=num_classes)
        #accuracy_per_epoch[e] = val_acc

        if use_scheduler and scheduler is not None:
            scheduler.step()

        train_acc.append(train_acc_e)
        val_acc.append(val_acc_e)
        print(f"Training acc: {train_acc_e:.3f}")
        print(f"Validation acc: {val_acc_e:.3f}")

        #torch.save(model.state_dict(), "output/model_weights_2b.pth")
        #print("Saved PyTorch Model State to output/model_weights_2b.pth")
        #torch.save(optimizer.state_dict(), "output/optimizer_state_2b.pth")
        #print("Saved PyTorch Optimizer State to output/optimizer_weights_2b.pth")
        if val_acc_e > best_acc_run:
            best_acc_run = val_acc_e
            #save_model(e, model, optimizer, criterion, best_acc_run, output_path)


        if val_acc_e >= best_accuracy:
            save_model(epochs=e, model=model, optimizer=optimizer, criterion=criterion, best_accuracy=best_accuracy, output_path=best_model_output_path)

        save_model(epochs=e, model=model, optimizer=optimizer, criterion=criterion, best_accuracy=best_accuracy, output_path=output_path)

    # save the trained model weights for a final time
    mean_acc = sum(val_acc)/len(val_acc)
    save_model(epochs=epochs, model=model, optimizer=optimizer, criterion=criterion, best_accuracy=mean_acc, output_path=output_path)
    # save the loss and accuracy plots
    save_plots(plot_path, train_acc, val_acc)
    # print("Evaluation succeeded")
    #evaluate_model(accuracy_per_epoch=val_acc)

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
