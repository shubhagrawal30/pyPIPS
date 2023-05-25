import numpy as np
import tqdm
from datetime import datetime

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.keras.backend.clear_session()
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, PROGRESS_EPOCH=50):
        self.PROGRESS_EPOCH = PROGRESS_EPOCH
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.PROGRESS_EPOCH == 0:
            print(f"{datetime.now().strftime('%H:%M:%S')}, epoch {epoch}: ", end="")
            for key, val in logs.items():
                print(f"{key}: {val:.3f}", end= "\t")
            print()

import pyPIPS.datasets as datasets


class BayConvPIPS():
    def __init__(self, dataset: datasets.Dataset):
        self.input_shape = dataset.P_kzs.shape[1:] + (1,)
        self.ks = dataset.ks
        self.zs = dataset.zs
        self.map_shape = (len(self.ks), len(self.zs))
        self.trained_on = dataset.num_points
        self.parameters = dataset.parameters

        self.X_train = dataset.P_kzs
        self.Y_train = dataset.all_parameters

        self.output_shape = self.Y_train.shape[1]

        self.model = None

        self.generated = False
        self.compiled = False
        self.fitted = False

        self.kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /
                           tf.cast(self.trained_on, dtype=tf.float32))
        
        self.epochs = []
        self.history = []

    def generate(self, get_summary=True, n_conv=None, f_conv=None, kernel=None, \
                 n_dense=None, f_dense=None, activation=tf.nn.tanh):
        if n_conv is not None:
            assert len(f_conv) == n_conv, "define independently number of filters per conv layer"
            assert len(kernel) == n_conv, "define independently kernel size for each conv layer"
        if n_dense is not None:
            assert len(f_dense) == n_dense, "define independently number of filters per dense layer"
    
        if n_conv is None:
            y, x = self.map_shape
            if y < 10 and x < 10:
                n_conv = 2
                f_conv = [8, 16]
                kernel = [(2, 2), (2, 2)]
            else:
                n_conv = 4
                f_conv = [4, 8, 16, 32]
                kernel = [(2, 2), (2, 2), (2, 2), (2, 2)]

        if n_dense is None:
            n_dense = 3
            f_dense = [128, 64, 32]


        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape))
        for i in range(n_conv):
            self.model.add(tfp.layers.Convolution2DFlipout(filters = f_conv[i], \
                        kernel_size=kernel[i], padding='same', activation=activation, \
                        kernel_divergence_fn=self.kl_divergence_function))
            self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"))

        self.model.add(tf.keras.layers.Flatten())

        for i in range(n_dense):
            self.model.add(tfp.layers.DenseFlipout(units = f_dense[i], \
                        activation=activation, \
                        kernel_divergence_fn=self.kl_divergence_function))
        
        self.model.add(tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(self.output_shape)))
        self.model.add(tfp.layers.MultivariateNormalTriL(self.output_shape))

        self.generated = True
        self.nconv = n_conv
        self.fconv = f_conv
        self.kernel = kernel
        self.ndense = n_dense
        self.fdense = f_dense
        self.activation = activation

        if get_summary:
            self.model.summary()

    def compile(self, optimizer=None, loss=None, metrics=None):
        assert self.generated, "model must be generated before compiling"
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()
        if loss is None:
            loss = lambda y_true, y_pred: -tf.reduce_mean(y_pred.log_prob(y_true))
        if metrics is None:
            metrics = ["mse", "mae"]
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.compiled = True

    def fit(self, val_dataset=None, epochs=20, verbose=1, callback_epoch=None):
        assert self.compiled, "model must be compiled before fitting"
        if val_dataset is None:
            validation_data = None
        else:
            validation_data = (val_dataset.P_kzs, val_dataset.all_parameters)
        if callback_epoch is None:
            self.history.append(self.model.fit(self.X_train, self.Y_train, epochs=epochs, \
                                verbose=verbose, validation_data=validation_data))
        else:
            self.history.append(self.model.fit(self.X_train, self.Y_train, epochs=epochs, \
                                verbose=verbose, callbacks=[CustomCallback(callback_epoch)], \
                                validation_data=validation_data))
        self.epochs.append(epochs)
        self.fitted = True

    def predict(self, dataset: datasets.Dataset, verbose=1):
        assert self.fitted, "model must be fitted before predicting"
        return self.model.predict(dataset.P_kzs, verbose=verbose)
    
    def evaluate(self, dataset: datasets.Dataset, verbose=1):
        assert self.fitted, "model must be fitted before evaluating"
        return self.model.evaluate(dataset.P_kzs, dataset.all_parameters, verbose=verbose)
