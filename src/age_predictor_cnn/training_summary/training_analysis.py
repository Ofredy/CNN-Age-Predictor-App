# Library imports
import matplotlib.pyplot as plt
import numpy as np


# File constants
TRAINING_SUMMARY_PATH = "loss_mae_summary.txt" 
TRAIN_SUM_OUTPUT_PLOT_PATH = "loss_mae_summary_plot.jpg"


if __name__ == "__main__":

    with open(TRAINING_SUMMARY_PATH, "r") as input_file:

        lines = input_file.readlines()

        num_epochs = len(lines) - 1

        epochs = np.zeros(num_epochs)
        train_loss = np.zeros(num_epochs)
        val_loss = np.zeros(num_epochs)
        val_age_mae = np.zeros(num_epochs)

        for idx, line in enumerate(lines):

            if idx == 0:
                continue

            line = line.split(',')
            epochs[idx-1] = idx 
            train_loss[idx-1] = float(line[0])
            val_loss[idx-1] = float(line[1])
            val_age_mae[idx-1] = float(line[2])

        input_file.close()

    # Plotting training, validation loss, and val_age_mae
    figure, axis = plt.subplots(1, 2)
    
    # Train and Val vs Epoch
    axis[0].plot(epochs, train_loss, label='Train')
    axis[0].plot(epochs, val_loss, label='Val')
    axis[0].set_xlabel("Epoch")
    axis[0].set_ylabel("Loss")
    axis[0].set_title("Train and Val Loss vs Epochs")
    axis[0].legend(loc='upper right')

    # Val AGE MAE vs Epoch
    axis[1].plot(epochs, val_age_mae)
    axis[1].set_xlabel("Epoch")
    axis[1].set_ylabel("Val Age MAE")
    axis[1].set_title("Val Age MAE vs Epochs")

    plt.savefig(TRAIN_SUM_OUTPUT_PLOT_PATH)
