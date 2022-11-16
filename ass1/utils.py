from random import random

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

""" 
util.py 
https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
"""

plt.style.use('ggplot')

def save_model(epochs, model, optimizer, criterion, best_accuracy, output_path):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving model ...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'accuracy': best_accuracy,
                }, output_path)

def load_model(e, model, optimizer, criterion, accuracy, output_path):
    """
    Function to load the trained model to disk.
    """
    print(f"Loading model ...")

    try:
        checkpoint = torch.load(output_path)
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

def find_better_model(output_path_1, model_name_1, output_path_2, model_name_2):
    _, _, _, _, accuracy_1 = load_model(e=None, model=None, optimizer=None, criterion=None,
                                             accuracy=0.0, output_path=output_path_1)
    _, _, _, _, accuracy_2 = load_model(e=None, model=None, optimizer=None, criterion=None,
                                             accuracy=0.0, output_path=output_path_2)
    best_model = model_name_2
    if accuracy_1 > accuracy_2:
        best_model = model_name_1
    return best_model


def save_plots(plot_path, train_acc, valid_acc, train_loss = None, valid_loss = None):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(plot_path)

    ## loss plots
    #plt.figure(figsize=(10, 7))
    #plt.plot(
    #    train_loss, color='orange', linestyle='-',
    #    label='train loss'
    #)
    #plt.plot(
    #    valid_loss, color='red', linestyle='-',
    #    label='validataion loss'
    #)
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.savefig('outputs/loss.png')

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
'''

def load_best_accuracy(best_accuracy_path):
    """
        Function to load the best accuracy in a cvs file.
    """
    with open(best_accuracy_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
             best_accuracy = row
    return best_accuracy

def save_best_accuracy(best_accuracy_path, best_accuracy):
    """
        Function to save the best accuracy in a cvs file.
    """
    #os.remove(best_accuracy_path)
    #with open(best_accuracy_path, 'w', newline='') as f:
    #    writer = csv.writer(f)
    #    writer.writerows(val_acc_e)

    np.savetxt(best_accuracy_path, best_accuracy)
    df = pd.DataFrame(best_accuracy)
    df.to_csv(os.path.join(output_path, output_file_path))
    
'''