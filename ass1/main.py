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

def data_preloading(dataset_path):
    subdir = sorted(glob(os.path.join(dataset_path, "*")))
    print(subdir)
    label = {}
    image_list = {}
    data_dict = {}
    data_dict_splitted = {}
    idx = 0
    idy = 0
    for sub in subdir:
        label[idx] = sub.split("/")[-1]
        print(label[idx])

        image_list[idx] = sorted(glob(os.path.join(sub, "*.jpg")))
        print(image_list[idx])
        print(len(image_list[idx]))
        data_dict[idx] = {"label": sub.split("/")[-1], "img_dir": sorted(glob(os.path.join(sub, "*.jpg")))}
        idx += 1
        for image_dir in sorted(glob(os.path.join(sub, "*.jpg"))):
            data_dict_splitted[idy] = {"label": sub.split("/")[-1], "img_dir": image_dir}
            idy += 1
    print(len(image_list))
    print("data_dict")
    print(data_dict)
    print("data_dict_splitted")
    print(data_dict_splitted)


    return

    # Press the green button in the gutter to run the script.
if __name__ == '__main__':

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # create dataset with custom Dataset class ImageDataset
    dataset_path = "/home/mmc-user/dataset_simpsons/imgs"
    data_preloading(dataset_path)

    # TODO implement training for different nets





# See PyCharm help at https://www.jetbrains.com/help/pycharm/

