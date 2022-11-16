import random as random
import numpy as np
from matplotlib import pyplot as plt
import torch
#from torch.utils.tensorboard import SummaryWriter

""" 
util.py includes different types of saving and loading functions. 
Also it includes a function to make experiments reproducible. 
Inspired by: https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
"""

plt.style.use('ggplot')

def save_model(epochs, model, optimizer, criterion, accuracy, name):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving model ...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'accuracy': accuracy,
                }, f"output/state_{name}.pth")

def load_model(name, e=None, model=None, optimizer=None, criterion=None, accuracy=None):
    """
    Function to load the trained model to disk.
    """
    print(f"Loading model ...")

    try:
        checkpoint = torch.load(f"output/state_{name}.pth")
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        e = checkpoint['epoch']
        criterion = checkpoint['loss']
        accuracy = checkpoint['accuracy']

        #best_checkpoint = torch.load("output/best_model_states.pth")
        #best_accuracy = best_checkpoint['best_accuracy']
    except FileNotFoundError:
        pass
    return e,  model, optimizer, criterion, accuracy

def find_better_model(model_name_1, model_name_2, name_1, name_2):
    _, _, _, _, accuracy_1 = load_model(name=name_1, e=None, model=None, optimizer=None, criterion=None, accuracy=0.0)
    _, _, _, _, accuracy_2 = load_model(name=name_2, e=None, model=None, optimizer=None, criterion=None, accuracy=0.0)
    best_model = model_name_2
    if accuracy_1 > accuracy_2:
        best_model = model_name_1
    return best_model

def save_plots(name, train_acc=None, valid_acc=None, train_loss=None, valid_loss=None):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    if train_acc or valid_loss is not None:
        # Initialise the subplot function using number of rows and columns
        plt.figure(figsize=(10, 7))
        if train_acc is not None:
            plt.plot(
                train_acc, color='green', linestyle='-',
                label='train accuracy'
            )
        if valid_acc is not None:
            plt.plot(
                valid_acc, color='blue', linestyle='-',
                label='validation accuracy'
            )
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f"output/accuracy_plot_{name}.png")

    ## loss plots
    if train_loss or valid_loss is not None:
        plt.figure(figsize=(10, 7))
        if train_loss is not None:
            plt.plot(
                train_loss, color='orange', linestyle='-',
                label='train loss'
            )
        if valid_loss is not None:
            plt.plot(
                valid_loss, color='red', linestyle='-',
                label='validation loss'
            )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"output/loss_plot_{name}.png")

def make_reproducible():
    """Set seeds to make the experiment reproducible (see https://pytorch.org/docs/stable/notes/randomness.html)"""
    torch.Generator().manual_seed(0)
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return worker_seed
"""

def evaluate_model(accuracy_per_epoch):
    print("Writing output")
    writer = SummaryWriter()

    for epoch in range(0, len(accuracy_per_epoch) ):
        writer.add_scalar('mean per class accuracy - validation', accuracy_per_epoch, epoch)

    writer.close()
"""

