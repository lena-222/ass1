import torch
import os
from glob import glob

import main
from experiments.experiment import model_train_and_eval
from experiments.experiment import make_reproducible

REMOTE_DATA_PATH = "/home/mmc-user/dataset_simpsons/imgs"
BATCH_SIZE = 64


def print_cuda_available(name):
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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # for 1.f)
    make_reproducible()

    # Implement machine learning things here.
    # print(cfg)

    full_dataset = ImageDataset("data/initial_data/test/labels_middle_2022-09-01-04-37-47.json",
                                "data/initial_data/test/", transform(True), transform_geometric(False),
                                transform_with_colorjitter())
    train_size = int(0.6 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    # train_data.dataset = copy.copy(full_dataset)  # is needed to work - still only train data is used

    # batchsize of 64
    batch_size = 64
    num_classes = 37
    epochs = 30
    learning_rate = 0.0001

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=1)

    accuracy_per_epoch = []

    # TODO implement training for different nets
    # TODO train the first ResNet
    # ResNet-training for 2a)
    model_train_and_eval(epochs, learning_rate=learning_rate,
                         model_name="Resnet18", accuracy_per_epoch=accuracy_per_epoch)
    # ConvNext-training for 2b)
    experiments.train_and_eval(epochs, learning_rate=learning_rate,
                               model_name="ConvNext", accuracy_per_epoch=accuracy_per_epoch)


    # Training of the best model with EMA-rate 2c)
    ema = True
    ema_rate = 0.998
    model_train_and_eval(epochs,learning_rate=learning_rate,
                         model_name="Resnet18", accuracy_per_epoch=accuracy_per_epoch)

    # add learning rate scheduler 2d)
    model_train_and_eval(epochs, learning_rate=learning_rate,
                         model_name="Resnet18", accuracy_per_epoch=accuracy_per_epoch)

    # add data augmentation 2e)

    model_train_and_eval(epochs, learning_rate=learning_rate,
                         model_name="Resnet18", accuracy_per_epoch=accuracy_per_epoch)

    # add more data augmentation 2f)
    model_train_and_eval(epochs, learning_rate=learning_rate,
                         model_name="Resnet18", accuracy_per_epoch=accuracy_per_epoch)


