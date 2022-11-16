"""! brief Experiment."""
import copy
from datetime import datetime

import torch
import torch.backends
import torch.backends.cudnn
import torch.cuda.amp as amp  # will be used for mixed precision training
import torch.nn as nn

from tqdm import tqdm  # will be used for visualization loading

import torch.utils.data
from torch.utils.data import Subset

from helper import get_mean_per_class, calculate_ema
from image_dataset import ImageDataset

# imports from my project
from pytorch_models import ConvNextTiny
from pytorch_models import ResNet18

from utils import save_plots, save_model, load_model, make_reproducible  #,
                # evaluate_model, load_best_accuracy, save_best_accuracy

def train(dataloader,
          model,
          optimizer,
          device,
          criterion,
          cur_iter,
          use_scaler,
          num_classes,
          ema=False,
          ema_rate=0.0,
          ema_model=None,
          testmod=False):
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

        if testmod:
            if counter == 3 :
                print("Training of epoch " + str(cur_iter) + " completed.")
                return get_mean_per_class(correct_labels=correct_labels, all_labels=all_labels, num_classes=num_classes)

        images = data[0].to(device)
        labels = data[1].to(device)

        optimizer.zero_grad()

        # torch.backends.cudnn.enabled = False

        if use_scaler:
            scaler = amp.GradScaler()
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        if ema:
            # calculate ema parameters
            calculate_ema(iter(model.parameters()), iter(ema_model.parameters()), ema_rate)
            calculate_ema(iter(model.buffers()), iter(ema_model.buffers()), ema_rate)

        # use mean per class accuracy as evaluation metric on train data
        for label, prediction in zip(labels, output):
            pred_label = torch.argmax(prediction)

            if pred_label == label:
                correct_labels[label] += 1

            all_labels[label] += 1

    # torch.cuda.synchronize()

    print("Training of epoch " + str(cur_iter) + " completed.")
    return get_mean_per_class(correct_labels=correct_labels, all_labels=all_labels, num_classes=num_classes)


def val(dataloader,
        model,
        device,
        cur_iter,
        num_classes,
        ema=False,
        testmod=False):

    """Evaluate model on validation data."""
    # model.train()
    # val_loss_list = []
    if ema:
        print('Validation of Ema')
    else:
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
            if testmod:

                if counter == 3:

                    print("Validation of epoch " + str(cur_iter) + " completed.")
                    return get_mean_per_class(correct_labels=correct_labels, all_labels=all_labels, num_classes=num_classes)

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

            # torch.cuda.synchronize()

    print("Validation of epoch " + str(cur_iter) + " completed.")
    return get_mean_per_class(correct_labels=correct_labels, all_labels=all_labels, num_classes=num_classes)


def model_train_and_eval(name,
                         dataset_path,
                         transform_type,
                         split_factor,
                         model_name,
                         batch_size,
                         num_classes,
                         epochs,
                         learning_rate,
                         num_workers=16,
                         use_scheduler=False,
                         ema=False,
                         ema_rate=None,
                         load=True):

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

    #load_data(dataset_path=dataset_path, transform_type=transform_type, split_factor=split_factor)
    # create dataset with custom Dataset class ImageDataset
    print("Start loading ImageDataSet...")
    train_dataset = ImageDataset(data_path=dataset_path, transform_type=transform_type)
    val_dataset = ImageDataset(data_path=dataset_path, transform_type="transform")

    # Split data into train and validation data
    print("Split dataset into train and validation set...")

    # get the training dataset size, need this to calculate the...
    # number if validation images
    # validation size = number of validation images
    valid_size = int((1 - split_factor) * len(train_dataset))
    # all the indices from the training set
    indices = torch.randperm(len(train_dataset)).tolist()
    train_data = Subset(train_dataset, indices[:-valid_size])
    val_data = Subset(val_dataset, indices[-valid_size:])
    print(f"Total training images: {len(train_data)}")
    print(f"Total validation images: {len(val_data)}")

    torch.Generator().manual_seed(0)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, num_workers=num_workers,
                                                   shuffle=True, generator=torch.Generator())
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True, drop_last=True, num_workers=num_workers,
                                                 generator=torch.Generator())

    # set model
    if model_name == 'ResNet18':
        model = ResNet18().to(device)
    elif model_name == 'ConvNext':
        model = ConvNextTiny().to(device)
    else:
        raise ValueError("Model not found!")
    print("Used Model: ", model_name)

    if ema:
        ema_model = copy.deepcopy(model)
        ema_model.to(device)
    else:
        ema_model = None

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
    ema_val_acc = []
    #best_acc_run = 0.0
    best_accuracy = 0.0


    print("Start training...")

    #start_e = 1
    for e in range(1, epochs + 1):
        if load:
            try:
                e, model, optimizer, criterion, best_acc_run = load_model(name=name, e=e, model=model, optimizer=optimizer,
                                                                          criterion=criterion)
                if ema:
                    _, ema_model, _, _, ema_best_acc_run = load_model(name=f'{name}_ema', e=e, model=ema_model, optimizer=optimizer,
                                                                  criterion=criterion)
            except FileNotFoundError:
                pass

        print("Current epoch: ", e)
        train_acc_e = train(dataloader=train_dataloader, model=model, optimizer=optimizer, device=device, criterion=criterion,
                            cur_iter=e, use_scaler=True, num_classes=num_classes, ema=ema, ema_rate=ema_rate, ema_model=ema_model)

        val_acc_e = val(dataloader=val_dataloader, model=model,
                      device=device, cur_iter=e, num_classes=num_classes)
        train_acc.append(train_acc_e)
        val_acc.append(val_acc_e)

        # if val_acc_e > best_acc_run:
        #    best_acc_run = val_acc_e

        mean_acc = sum(val_acc) / len(val_acc)
        save_model(epochs=e, model=model, optimizer=optimizer, criterion=criterion, accuracy=mean_acc, name=name)

        if ema:
            ema_val_acc_e = val(dataloader=val_dataloader, model=ema_model,
                                device=device,  cur_iter=e, num_classes=num_classes,ema=ema)
            ema_val_acc.append(ema_val_acc_e)

            print(f"Training acc: {train_acc_e:.3f}")
            print(f"Validation acc: {val_acc_e:.3f}, ema validation acc: {ema_val_acc_e:.3f}")

            ema_mean_acc = sum(ema_val_acc) / len(ema_val_acc)

            save_model(epochs=e, model=ema_model, optimizer=optimizer, criterion=criterion, accuracy=ema_mean_acc,
                       name=f"{name}_ema")
        else:
            print(f"Training acc: {train_acc_e:.3f}")
            print(f"Validation acc: {val_acc_e:.3f}")

        if use_scheduler and scheduler is not None:
            scheduler.step()

    # save the trained model weights for a final time
    save_model(epochs=epochs, model=model, optimizer=optimizer, criterion=criterion, accuracy=mean_acc, name=name)

    # save the loss and accuracy plots
    save_plots(name, train_acc, val_acc)

    # Evaluation with tensorboard,
    # evaluate_model(accuracy_per_epoch=val_acc)
    # print("Evaluation succeeded")

    # compare trained model and best model
    try:
        _, _, _, _, best_accuracy = load_model(name="best", e=None, model=None, optimizer=None, criterion=criterion,
                                               accuracy=best_accuracy)
    except FileNotFoundError:
        pass
    # save the best model
    if mean_acc >= best_accuracy:
        save_model(epochs=epochs, model=model, optimizer=optimizer, criterion=criterion, accuracy=best_accuracy,
                   name="best")

    print("Done!")
