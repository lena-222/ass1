"""! @brief Experiment."""

# import signal
import torch
import copy
from sacred import Experiment
from ema_pytorch import EMA
import numpy as np
import random
# from util.general_utils import signal_handler
from models.pytorch_models import ResNet18
from models.ImageDataset import ImageDataset
from models.ImageDataset import transform
from models.ImageDataset import transform_geometric
from models.ImageDataset import transform_with_colorjitter




ex = Experiment()

# make experiment reproducible with notations from:
# https://pytorch.org/docs/stable/notes/randomness.html
def make_reproducible():
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True

def train(dataloader, model, optimizer, device, cur_iter, scaler=None):
    """Train model one epoch."""

    model.train()
    for batch, (images, targets) in enumerate(dataloader):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # TODO add Mixed Precision Training
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
            #loss = loss_function()

    val_loss_mean = np.mean(val_loss_list)
    print("val loss: ", val_loss_mean)
    if bol_log:
        ex.log_scalar("validation.loss", val_loss_mean, cur_iter)

    return val_loss_mean


def get_test_accuracy(model, val_dataloader, device):
    pass


def model_train_and_eval(model, train_dataloader, val_dataloader, train_data, batch_size, num_classes, epochs, learning_rate,  device):

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    criterion = torch.nn.CrossEntropyLoss()
    # use of adam optimizer and an learning rate of 0.0001
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    # set 30 epochs of training
    num_batches = np.ceil(len(train_data) / batch_size)
    best_acc = 0
    for e in range(epochs):
        cur_iter = e * num_batches
        train(train_dataloader, model, optimizer, device, cur_iter, scaler=None)
        cur_iter += num_batches
        val_acc, _, _ = get_test_accuracy(model, val_dataloader, device)
        val(val_dataloader, model, device, cur_iter)

        if val_acc >= best_acc:
            best_acc = val_acc
        

    torch.save(model.state_dict(), "output/model_weights.pth")
    model.load_state_dict(torch.load('model_weights.pth'))
    print("Saved PyTorch Model State to output/model_weights.pth")

    print("Done!")


@ex.main
def run(_config):
    """Register signal handler."""
    # signal.signal(signal.SIGINT, signal_handler)
    # cfg = _config

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # for 1.f)
    make_reproducible()

    # Implement machine learning things here.
    # print(cfg)

    full_dataset = ImageDataset("data/initial_data/test/labels_middle_2022-09-01-04-37-47.json",
                                "data/initial_data/test/", transform(True), transform_geometric(False), transform_with_colorjitter())
    train_size = int(0.6 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_data.dataset = copy.copy(full_dataset)  # is needed to work - still only train data is used

    # batchsize of 64
    batch_size = 64

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1)

    # TODO num_classes
    num_classes = 37
    epochs = 30
    learning_rate = 0.0001

    model = ResNet18().model.to(device)


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