from experiment import model_train_and_eval

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    dataset_path = "/home/mmc-user/dataset_simpsons/imgs"

    # Implement machine learning things here.

    # parameters for training
    batch_size = 64
    num_classes = 37
    epochs = 30
    learning_rate = 0.0001

    transform_type = "transform"
    '''
    # TODO implement training for different nets
    # TODO train the first ResNet
    # ResNet-training for 2a)
    model_train_and_eval(dataset_path="/home/mmc-user/dataset_simpsons/imgs",
                         transform_type=transform_type,
                         model_name="ResNet18",
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate)
    '''

    # ConvNext-training for 2b)
    model_train_and_eval(dataset_path=dataset_path,
                         transform_type=transform_type,
                         model_name="ConvNext",
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate)


    '''
    # Training of the best model with EMA-rate 2c)
    ema = True
    ema_rate = 0.998
    model_train_and_eval(dataset_path=dataset_path,
                         transform_type=transform_type,
                         model_name="ConvNext",
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate)
    '''

    '''
    # add learning rate scheduler 2d)
    model_train_and_eval(dataset_path=dataset_path,
                         transform_type=transform_type,
                         model_name="ConvNext",
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate)
    '''

    '''
    # add data augmentation 2e)

    model_train_and_eval(dataset_path=dataset_path,
                         transform_type="geometric",
                         model_name="ConvNext",
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate)
    '''

    '''
    # add more data augmentation 2f)

    model_train_and_eval(dataset_path=dataset_path,
                         transform_type="colorjitter",
                         model_name="ConvNext",
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate)
    '''
