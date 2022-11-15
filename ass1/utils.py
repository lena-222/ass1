from matplotlib import pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import csv
import os
import numpy as np
import pandas as pd

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