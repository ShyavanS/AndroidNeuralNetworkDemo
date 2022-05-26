import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

data = datasets.load_diabetes(as_frame=True)['frame']

diabetes_y = data.target
diabetes_x = data.drop(['target'], axis=1)

train_x, test_x, train_y, test_y = train_test_split(
    diabetes_x, diabetes_y, test_size=0.2, random_state=1)
train_x, data_x, train_y, data_y = train_test_split(
    train_x, train_y, test_size=0.5, random_state=1)

data_holdover = pd.merge(data_x, data_y, left_index=True, right_index=True)

model = models.Sequential()

model.add(layers.Dense(10, input_dim=10,
          kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(20, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(15, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(10, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(15, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(5, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(1, kernel_initializer='normal'))


def train_model(epoch, rate):
    global history, prediction, history_dict

    model.compile(loss='mean_squared_error',
                  optimizer=tf.optimizers.Adam(learning_rate=rate))

    history = model.fit(train_x, train_y, epochs=epoch,
                        batch_size=10, validation_split=0.2)
    history_dict = history.history
    prediction = model.predict(test_x)
    rms = mean_squared_error(test_y, prediction, squared=False)

    return rms


def feed_data(arr_x, arr_y):
    global train_x, train_y

    arr_y = [int(i) for i in arr_y]

    arr_x = [[float(j) for j in i] for i in arr_x]

    arr_x = pd.DataFrame(arr_x, columns=train_x.columns)
    arr_y = pd.Series(arr_y)

    train_x = train_x.append(arr_x, ignore_index=True)
    train_y = train_y.append(arr_y, ignore_index=True)


def random_select():
    global train_x, train_y, data_holdover

    data_holdover = data_holdover.reset_index(drop=True)

    selection = data_holdover.sample()

    data_holdover = data_holdover.drop(data_holdover.index[selection.index])

    selection_y = selection.target
    selection_x = selection.drop(['target'], axis=1)

    train_x = train_x.append(selection_x, ignore_index=True)
    train_y = train_y.append(selection_y, ignore_index=True)

    if data_holdover.empty:
        return True

    return False


def plot_scatter():
    p1 = max(max(prediction), max(test_y))
    p2 = min(min(prediction), min(test_y))

    plt.figure()

    plt.scatter(test_y, prediction)
    plt.plot([p1, p2], [p1, p2], 'b-')

    plt.title('Model Scatter')
    plt.ylabel('Predictions')
    plt.xlabel('Actual Values')
    plt.yscale('log')
    plt.xscale('log')

    plt.savefig(os.path.join(os.environ['HOME'], 'scatter.png'))


def plot_loss():
    plt.figure()

    plt.plot(history_dict['loss'])
    plt.plot(history_dict['val_loss'])

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.savefig(os.path.join(os.environ['HOME'], 'loss.png'))


def save(dirname):
    model.save(os.path.join(os.environ['HOME'], f'{dirname}'))
    with open(os.path.join(os.environ['HOME'], f'{dirname}.json'), 'w') as stdout:
        NEURAL_DICT = {
            "history": history_dict,
            "prediction": prediction.tolist()
        }
        json.dump(NEURAL_DICT, stdout, indent=4)


def load(dirname):
    global model, history_dict, prediction

    model = models.load_model(os.path.join(os.environ['HOME'], f'{dirname}'))
    with open(os.path.join(os.environ['HOME'], f'{dirname}.json'), 'r') as stdin:
        NEURAL_DICT = json.load(stdin)
        history_dict = NEURAL_DICT['history']
        prediction = np.array(NEURAL_DICT['prediction'])


if __name__ == "__main__":
    print(train_model(500, 0.01))
