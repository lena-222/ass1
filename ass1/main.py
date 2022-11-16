from experiment import model_train_and_eval
from utils import find_better_model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    dataset_path = "/home/mmc-user/dataset_simpsons/imgs"

    # Implement machine learning things here.

    # parameters for training
    batch_size = 64
    num_classes = 37
    epochs = 30
    learning_rate = 0.0001
    split_factor = 0.6

    transform_type = "transform"

    best_model_output_path = "output/best_model_states.pth"

    # TODO implement training for different nets
    # TODO train and implement the ema net

    # ResNet-training for 2a)
    model_train_and_eval(name="2a",
                         dataset_path=dataset_path,
                         transform_type=transform_type,
                         split_factor=split_factor,
                         model_name="ResNet18",
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         num_workers=16,
                         use_scheduler=False,
                         load=False)

    # ConvNext-training for 2b)
    model_train_and_eval(name="2b",
                         dataset_path=dataset_path,
                         transform_type=transform_type,
                         split_factor=split_factor,
                         model_name="ConvNext",
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         num_workers=16,
                         use_scheduler=False,
                         load=False)

    model_name = find_better_model(model_name_1="ResNet18", model_name_2="ConvNext",
                                   name_1="2a", name_2="2b")
    print("Best model so far:", model_name)
    # Training of the best model with EMA-rate 2c)

    model_train_and_eval(name="2c",
                         dataset_path=dataset_path,
                         transform_type=transform_type,
                         split_factor=split_factor,
                         model_name=model_name,
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate, 
                         num_workers=16,
                         ema=True,
                         ema_rate=0.998)


    '''
    # add learning rate scheduler 2d)
    model_train_and_eval(name="2d",
                         dataset_path=dataset_path,
                         transform_type=transform_type,
                         split_factor=split_factor,
                         model_name=model_name,
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         num_workers=16,
                         use_scheduler=True)
    '''

    '''
    # add data augmentation 2e)

    model_train_and_eval(name="2e",
                         dataset_path=dataset_path,
                         transform_type=model_name,
                         split_factor=split_factor,
                         model_name="ConvNext",
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         num_workers=16,
                         learning_rate=learning_rate)
    '''

    '''
    # add more data augmentation 2f)

    model_train_and_eval(name="2f",
                         dataset_path=dataset_path,
                         transform_type=model_name,
                         split_factor=split_factor,
                         model_name=best_model,
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate)
    '''
