"""! @brief Experiment."""

# import signal
import sys

import torch
import torch.backends
#from ema_pytorch import EMA
import numpy as np
import random
import torch.cuda.amp as amp # will be used for mixed precision training
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # will be used for visualization loading

# from util.general_utils import signal_handler
from pytorch_models import ResNet18
from pytorch_models import ConvNextTiny


# make experiment reproducible with notations from:
# https://pytorch.org/docs/stable/notes/randomness.html
def make_reproducible():
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.Generator().manual_seed(0)
    #torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True

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
        
        for batch, data in enumerate(dataloader):
            #print("batch")
            #print(batch)
            #print("images")
            #print(x)
            #print("labels")
            #print(y)
            #input = images.to(device)
            #images = list(image.to(device) for image in x)
            # labels = list(label.to(device) for label in y)
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            #for i, data in enumerate(training_loader, 0):
            #    inputs, labels = data[0].to(device), data[1].to(device)

            #torch.backends.cudnn.deterministic = False
            #images, labels = data.to(device)
            images = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()

            # torch.backends.cudnn.enabled = False
            if use_scaler:
                scaler = amp.GradScaler()
                # TODO add Mixed Precision Training
                with torch.cuda.amp.autocast(enabled=scaler is not None):

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
            #torch.cuda.synchronize()
            # TODO Wofür ist new_iter?
            new_iter = cur_iter + batch


def val(dataloader,
        data,
        model,
        device,
        cur_iter,
        num_classes,
        batch_size):
    """Evaluate model on validation data."""
    #model.train()
    #val_loss_list = []

    all_labels = torch.zeros(model.out_features)
    correct_labels = torch.zeros(model.out_features)

    model.eval()

    scaler = amp.GradScaler()

    with torch.no_grad():

        with tqdm(total=len(data), desc="Validation progress:\t", bar_format="{l_bar}{bar:50}| \\ [ epoch " +
                                                                                        str(cur_iter) + " ]",
                  file=sys.stdout) as pbar:

            for batch, data in enumerate(dataloader):
                images = data[0].to(device)
                labels = data[1].to(device)
                #images = list(image.to(device) for image in images)
                #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                #labels = list(label.to(device) for label in labels)
                output = model(images)
                
                # use mean per class accuracy as evaluation metric
                for label, prediction in zip(labels, output):
                    pred_label = torch.argmax(prediction)
                    
                    if pred_label == label:
                        correct_labels[label] += 1
                    
                    all_labels[label] += 1

                pbar.update(batch_size)
                #torch.cuda.synchronize()

    # mean per class accuracy
    not_zero = []
    for i in range(0, num_classes):
        if all_labels[i] > 0:
            not_zero.append(i)

    numerator = [correct_labels[i] / all_labels[i] for i in not_zero]
    accuracy = sum(numerator) / len(not_zero)

    print("Medium per class accuracy: " + str(accuracy.item()))

    return accuracy

def model_train_and_eval(model_name,
                         train_dataloader,
                         val_dataloader,
                         train_data,
                         val_data,
                         batch_size,
                         num_classes,
                         epochs,
                         learning_rate,
                         device,
                         accuracy_per_epoch,
                         use_scheduler = False,
                         ema = False,
                         ema_rate = None):

    if model_name == 'ResNet18':
        model = ResNet18().to(device)
    elif model_name == 'ConvNext':
        model = ConvNextTiny().to(device)
    else:
        raise ValueError("Model not found!")

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    # criterion for classification
    criterion = torch.nn.CrossEntropyLoss()

    # set adam optimizer and a learning rate of 0.0001
    # eventually create schedular
    if use_scheduler:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    else:
        optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)

    num_batches = np.ceil(len(train_data) / batch_size)
    best_acc = 0 # save the best accuracy later
    use_scaler = False
    for e in range(epochs):
        cur_iter = e * num_batches
        with tqdm(total=len(train_data), desc="Training progress:\t\t",
                  bar_format="{l_bar}{bar:50}| \\ [ epoch " + str(cur_iter) +
                             " ]",file=sys.stdout) as pbar:
            train(dataloader=train_dataloader, data=train_data,
                  model=model, optimizer=optimizer, device=device, criterion=criterion,
                  cur_iter=cur_iter, batch_size=batch_size, use_scaler=use_scaler)
            pbar.update(batch_size)
            cur_iter += num_batches
            #val_acc, _, _ = get_test_accuracy(model, val_dataloader, device)

            val_acc = val(dataloader=val_dataloader, data=val_data, model=model,
                          device=device, cur_iter=cur_iter, num_classes=num_classes, batch_size=batch_size).item()
            accuracy_per_epoch.append(val_acc)

            if val_acc >= max(accuracy_per_epoch):
                #best_acc = val_acc
                torch.save(model.state_dict(), "output/best_model_weights.pth")
                torch.save(optimizer.state_dict(), "output/best_optimizer_state.pth")
                print("Saved best PyTorch Model State to output/model_weights.pth")
            if use_scheduler:
                scheduler.step()

        torch.save(model.state_dict(), "output/model_weights_ConvNeXt_Tiny.pth")
        model.load_state_dict(torch.load('output/model_weightsConvNeXt_Tiny.pth'))
        print("Saved PyTorch Model State to output/model_weightsConvNeXt_Tiny.pth")
        torch.save(optimizer.state_dict(), "output/optimizer_stateConvNeXt_Tiny.pth")
        print("Saved PyTorch Optimizer State to output/optimizer_weightsConvNeXt_Tiny.pth")
        optimizer.load_state_dict(torch.load('output/optimizer_stateConvNeXt_Tiny.pth'))


    print("Writing output")
    writer = SummaryWriter()

    for epoch in range(0, len(accuracy_per_epoch)):
        writer.add_scalar('mean per class accuracy - validation', accuracy_per_epoch[epoch], epoch)

    writer.close()

    print("Done!")


# TODO use a cycle learning rate scheduler
# search for torch.optim.lr_schedular.CyclicLR
''' NOT USED YET 
https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR
Beispielcode zu: 

Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR). 
The policy cycles the learning rate between two boundaries with a constant frequency, as detailed in
the paper Cyclical Learning Rates for Training Neural Networks. The distance between the two boundaries 
can be scaled on a per-iteration or per-cycle basis.

Cyclical learning rate policy changes the learning rate after every batch. 
step should be called after a batch has been used for training.


optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
data_loader = torch.utils.data.DataLoader(...)
for epoch in range(10):
    for batch in data_loader:
        train_batch(...)
        scheduler.step()
'''
    
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