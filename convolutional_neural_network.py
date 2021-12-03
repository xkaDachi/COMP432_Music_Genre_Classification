import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

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

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.
    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split
    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """Generates CNN model
    :param input_shape (tuple): Shape of input set
    :return model: CNN model
    """

    # build network topology using sequential model
    # model = keras.Sequential()
    model = torch.nn.Sequential()

    #keras.layers.Conv2D = filters, kernel_size, strides=(1, 1), padding='valid',
    #         data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
    #         use_bias=True, kernel_initializer='glorot_uniform',
    #         bias_initializer='zeros', kernel_regularizer=None,
    #         bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    #         bias_constraint=None, **kwargs


    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    # 1st conv layer
    # kernels/filters= 32, kernel_size/grid_size=(3, 3), activation='relu', input_shape=input_shape)
    # model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(torch.nn.Conv2D(out_channels=32, kernel_size=(3, 3), padding='same'))
    model.add(torch.nn.ReLU()) # input_shape?


    # tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs)

    # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    # pool_size/grid_size=(3, 3), strides=(2, 2), padding='same' => zero padding?
    # model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(torch.nn.MaxPool2D(kernel_size=(3, 3), strides=(2, 2), padding='same'))


    # process that standardizes/normalizes the activations in the current layer
    # adv: speeds up training substanstially, converges faster + more reliable
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(torch.nn.Conv2D(out_channels=32, kernel_size=(3, 3), padding='same'))
    model.add(torch.nn.ReLU())
    model.add(torch.nn.MaxPool2D(kernel_size=(3, 3), strides=(2, 2), padding='same'))

    # 3rd conv layer
    model.add(torch.nn.Conv2D(out_channels=32, kernel_size=(3, 3), padding='same'))
    model.add(torch.nn.ReLU()) # input_shape?
    model.add(torch.nn.MaxPool2D(kernel_size=(3, 3), strides=(2, 2), padding='same'))


    # flatten output and feed it into dense layer
    # model.add(keras.layers.Flatten())
    mode.add(torch.nn.Flatten())

    # model.add(keras.layers.Dense(64, activation='relu')) 64 == neurons
    model.add(torch.nn.Linear(out_features=64))
    model.add(torch.nn.ReLU())

    # model.add(keras.layers.Dropout(0.3))
    model.add(torch.nn.Dropout(p=0.3)) #30% or 0.3 == rate for overfitting

    # output layer
    # model.add(keras.layers.Dense(10, activation='softmax')) 10 == neurons that is the num of genres used
    model.add(torch.nn.Linear(out_features=10)) #10 == number of outputs/number of genres used
    model.add(torch.nn.Softmax()) #dim=10???? A dimension along which Softmax will be computed (so every slice along dim will sum to 1).

    return model


def predict(model, X, y):
    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


def main():
    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # compile model
    optimiser = torch.optim.Adam(learning_rate=0.0001)
    # optimiser = torch.optim.Adam(learning_rate=0.0001) ??
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # pick a sample to predict from the test set
    X_to_predict = X_test[100]
    y_to_predict = y_test[100]

    # predict sample
    predict(model, X_to_predict, y_to_predict)


if __name__ == "__main__":
    main()
