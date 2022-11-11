import torch
import os
from glob import glob
import string
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# nvidia-smi
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    print(torch.cuda.is_available())

def data_loading():

    dataset_path = "/home/mmc-user/dataset_simpsons/imgs"
    subdir = sorted(glob(os.path.join(dataset_path, "*")))
    print(subdir)
    label = {}
    image_list = {}
    i = 0
    for sub in subdir:
        #label[i] = sub.split("/")[-1]
        #print(label[i])
        image_list[i] = sorted(glob(os.path.join(sub, "*.jpg")))
        print(image_list[i])
        print(len(image_list[i]))
        i += 1

    print(len(image_list))

    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_loading()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

