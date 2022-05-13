import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn import datasets

diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)

train_x, _, test_x = np.split(diabetes_x, [310, 310], 0)
train_y, _, test_y = np.split(diabetes_y, [310, 310], 0)

model = models.Sequential()

model.add(layers.Dense(10, input_dim=10,
          kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(20, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(15, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(10, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(5, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(1, kernel_initializer='normal'))


def train_model(epoch, rate):
    global history, prediction, history_dict

    model.compile(loss='mean_squared_error',
                  optimizer=tf.optimizers.Adam(learning_rate=rate))

    history = model.fit(train_x, train_y, epochs=epoch,
                        batch_size=10, validation_split=0.3)
    history_dict = history.history
    prediction = model.predict(test_x)

    rms = float(tf.sqrt(tf.reduce_mean((test_y - prediction) ** 2)))

    return rms


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
