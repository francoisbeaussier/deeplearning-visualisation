import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

tf.random.set_seed(1)
np.random.seed(1)

x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0

x_train = x[:100, :]
y_train = y[:100]
x_valid = x[100:, :]
y_valid = y[100:]

# fig = plt.figure(figsize=(6, 6))
# plt.plot(x[y==0, 0], x[y==0, 1], 'o', alpha=0.75, markersize=10)
# plt.plot(x[y==1, 0], x[y==1, 1], '^', alpha=0.75, markersize=10)
# plt.xlabel(r'$x_1$', size=15)
# plt.ylabel(r'$x_2$', size=15)
# plt.show()

model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Dense(
        units=4,
        input_shape=(2,),
        activation='relu'
    ))
model.add(
    tf.keras.layers.Dense(
        units=4,
        activation='relu'
    ))
model.add(
    tf.keras.layers.Dense(
        units=4,
        activation='relu'
    ))
model.add(
    tf.keras.layers.Dense(
        units=1,
        activation='sigmoid'
    ))
    
model.summary()

def plot_graph(model, epoch, history, x_valid, y_valid):
    if epoch == 0:
        return

    fig = plt.figure(figsize=(16, 4))

    ax = fig.add_subplot(1, 3, 1)
    plt.plot(history['loss'], lw=4)
    plt.plot(history['val_loss'], lw=4)
    plt.legend(['Train loss', 'Validation loss'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)

    ax = fig.add_subplot(1, 3, 2)
    plt.plot(history['binary_accuracy'], lw=4)
    plt.plot(history['val_binary_accuracy'], lw=4)
    plt.legend(['Train acc', 'Validation acc'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)

    ax = fig.add_subplot(1, 3, 3)
    plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer), clf=model, zoom_factor=1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

    ax.set_xlabel(r'$x_1$', size=15)
    ax.xaxis.set_label_coords(1, -0.025)
    ax.set_ylabel(r'$x_2$', size=15)
    ax.yaxis.set_label_coords(-0.025, 1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.9])

    fig.suptitle('Learning XOR (Tensorflow - 3 hidden layers of 4 units each)') 
    plt.savefig(f'xor\\training{epoch:>03d}.png')
    plt.show()

class PlotStep(tf.keras.callbacks.Callback):
    def __init__(self, model, x_valid, y_valid):
        super(PlotStep, self).__init__()
        self.model = model
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.logs = []

    def on_train_begin(self, logs={}):
        self.history = {}
        self.history['loss'] = []
        self.history['val_loss'] = []
        self.history['binary_accuracy'] = []
        self.history['val_binary_accuracy'] = []

        self.logs.append('on_train_begin')

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)

        print(logs)
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['binary_accuracy'].append(logs.get('binary_accuracy'))
        self.history['val_binary_accuracy'].append(logs.get('val_binary_accuracy'))
        
        #print('on_epoch_end!')
        plot_graph(self.model, epoch, self.history, self.x_valid, self.y_valid)
        
model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy()
    ])

hist = model.fit(
    x_train, 
    y_train, 
    validation_data=(x_valid, y_valid),
    callbacks=[PlotStep(model, x_valid, y_valid)],
    epochs=2,
    batch_size=2
)

from mlxtend.plotting import plot_decision_regions

history = hist.history

