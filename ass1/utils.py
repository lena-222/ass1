from matplotlib import pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

""" 
util.py 
https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
"""

plt.style.use('ggplot')

def save_model(epochs, model, optimizer, criterion, name):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/final_model.pth')

def save_plots(train_acc, valid_acc, train_loss = None, valid_loss = None):
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
    plt.savefig('outputs/accuracy.png')

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

def evaluate_model(accuracy_per_epoch):
    print("Writing output")
    writer = SummaryWriter()

    for epoch in range(0, len(accuracy_per_epoch)):
        writer.add_scalar('mean per class accuracy - validation', accuracy_per_epoch[epoch], epoch)

    writer.close()