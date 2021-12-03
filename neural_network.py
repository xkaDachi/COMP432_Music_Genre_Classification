import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch

# path to json file that stores MFCCs and genre labels for each processed segment
# DATA_PATH = "/data.json"
DATA_PATH = "./data.json"  # for windows


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error")

    plt.show()


if __name__ == "__main__":

    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build network topology
    model = torch.nn.Sequential([

        # input layer
        torch.nn.Flatten((X.shape[1], X.shape[2])),

        # 1st dense layer
        torch.nn.Linear(512, activation='relu',
                        kernel_regularizer=torch.norm(input=0.001, dtype=torch.float)),
        torch.nn.Dropout(0.3),

        # 2nd dense layer
        torch.nn.Linear(256, activation='relu',
                        kernel_regularizer=torch.norm(input=0.001, dtype=torch.float)),
        torch.nn.Dropout(0.3),

        # 3rd dense layer
        torch.nn.Linear(64, activation='relu',
                        kernel_regularizer=torch.norm(input=0.001, dtype=torch.float)),
        torch.nn.Dropout(0.3),

        # output layer
        torch.nn.Linear(10, activation='softmax')
    ])

    # compile model
    optimiser = torch.optim.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(
        X_test, y_test), batch_size=32, epochs=100)

    # plot accuracy and error as a function of the epochs
    plot_history(history)
