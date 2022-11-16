from experiment import model_train_and_eval

# Press the green button in the gutter to run the script.
from utils import load_model, find_better_model

if __name__ == '__main__':

    dataset_path = "/home/mmc-user/dataset_simpsons/imgs"

    # Implement machine learning things here.

    # parameters for training
    batch_size = 64
    num_classes = 37
    epochs = 30
    learning_rate = 0.0001

    transform_type = "transform"

    best_model_output_path = "output/best_model_states.pth"
    # TODO implement training for different nets
    # TODO train and implement the ema net
    '''
    # ResNet-training for 2a)
    model_train_and_eval(best_model_output_path=best_model_output_path,
                         plot_path="output/accuracy_plot_2b.png",
                         output_path="output/state_2a.pth",
                         dataset_path=dataset_path,
                         transform_type=transform_type,
                         model_name="ResNet18",
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         num_workers=16,
                         use_scheduler=False)

    # ConvNext-training for 2b)
    model_train_and_eval(plot_path="output/accuracy_plot_2b.png",
                         output_path="output/state_2b.pth",
                         dataset_path=dataset_path,
                         transform_type=transform_type,
                         model_name="ConvNext",
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         num_workers=16,
                         use_scheduler=False)
    '''

    model_name = find_better_model("output/state_2a.pth", "ResNet18", "output/state_2b.pth", "ConvNext")
    print(model_name)
    # Training of the best model with EMA-rate 2c)
    '''
    model_train_and_eval(dataset_path=dataset_path,
                         transform_type=transform_type,
                         model_name=model_name,
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate, 
                         num_workers=16,
                         ema=True,
                         ema_rate=0.998)

    '''
    '''
    # add learning rate scheduler 2d)
    model_train_and_eval(dataset_path=dataset_path,
                         transform_type=transform_type,
                         model_name="ConvNext",
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         learning_rate=learning_rate,
                         num_workers=16,
                         use_scheduler=True)
    '''

    '''
    # add data augmentation 2e)

    model_train_and_eval(dataset_path=dataset_path,
                         transform_type="geometric",
                         model_name="ConvNext",
                         batch_size=batch_size,
                         num_classes=num_classes,
                         epochs=epochs,
                         num_workers=16,
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
