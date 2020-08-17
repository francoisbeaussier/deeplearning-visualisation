import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions

tf.random.set_seed(1)
np.random.seed(1)

from matplotlib.colors import ListedColormap

class DecisionRegionPlot():
    def __init__(self, resolution):
        self.resolution = resolution
        self.initialized = False
    
    def setup(self, X, y):
        self.markers = ('s', 'x', 'o', '^', 'v')
        self.colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        self.cmap = ListedColormap(self.colors[:len(np.unique(y))])

        self.x1_min, self.x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        self.x2_min, self.x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        self.xx1, self.xx2 = np.meshgrid(
            np.arange(self.x1_min, self.x1_max, self.resolution), 
            np.arange(self.x2_min, self.x2_max, self.resolution))
        self.map = np.array([self.xx1.ravel(), self.xx2.ravel()]).T
        self.initialized = True
 
    def plot(self, X, y, clf):
        if self.initialized == False:
            self.setup(X, y)

        Z = clf.predict(self.map)
        Z = Z.reshape(self.xx1.shape)
        plt.contourf(self.xx1, self.xx2, Z, alpha=0.3, cmap=self.cmap)
        plt.xlim(self.xx1.min(), self.xx1.max())
        plt.ylim(self.xx2.min(), self.xx2.max())

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=self.colors[idx], marker=self.markers[idx], label=cl, edgecolor='black')

x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0

x_train = x[:100, :]
y_train = y[:100]
x_valid = x[100:, :]
y_valid = y[100:]

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

drp = DecisionRegionPlot(resolution=0.02)

def plot_graph(model, epoch, history, x_valid, y_valid):

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(12, 12)

    ax = fig.add_subplot(gs[:6, :6])
    plt.plot(history['loss'], lw=3)
    plt.plot(history['val_loss'], lw=3)
    plt.legend(['Train loss', 'Validation loss'], fontsize=15)
    # ax.set_xlabel('Epochs', size=15)

    ax = fig.add_subplot(gs[6:, :6])
    plt.plot(history['binary_accuracy'], lw=3)
    plt.plot(history['val_binary_accuracy'], lw=3)
    plt.legend(['Train acc', 'Validation acc'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)

    ax = fig.add_subplot(gs[:, 6:])
    drp.plot(x_valid, y_valid.astype(np.integer), model)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])

    ax.set_xlabel(r'$x_1$', size=15)
    # ax.xaxis.set_label_coords(1, -0.025)
    ax.set_ylabel(r'$x_2$', size=15)
    ax.yaxis.set_label_coords(0.05, 0.5)

    # plt.tight_layout(rect=[0, 0.03, 1, 0.9])

    fig.suptitle('Learning XOR with Tensorflow - 3 hidden layers of 4 units each', size=18)
    plt.savefig(f'xor\\training{epoch+1:>03d}.png')
    # plt.show()
    fig.clear()

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
    epochs=200,
    batch_size=2
)

# from mlxtend.plotting import plot_decision_regions

# history = hist.history

