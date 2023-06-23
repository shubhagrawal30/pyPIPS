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

from sklearn.preprocessing import StandardScaler

import pyPIPS.datasets as datasets

class PIPSModel():
    def __init__(self, dataset: datasets.Dataset):
        self.input_shape = dataset.P_kzs.shape[1:] + (1,)
        self.ks = dataset.ks
        self.zs = dataset.zs
        self.map_shape = (len(self.ks), len(self.zs))
        self.trained_on = dataset.num_points
        self.parameters = dataset.parameters

        self.X_train = dataset.P_kzs
        self.Y_train = dataset.all_parameters

        self.has_scaled = dataset.has_scaled
        self.scaler = dataset.scaler
        self.unscaled_parameters = dataset.unscaled_parameters

        self.output_shape = self.Y_train.shape[1]

        self.model = None

        self.generated = False
        self.compiled = False
        self.fitted = False
        
        self.epochs = []
        self.history = []      
    
    def compile(self):
        assert self.optimizer is not None, "optimizer must be set before compiling"
        assert self.loss is not None, "loss must be set before compiling"
        assert self.metrics is not None, "metrics must be set before compiling"
        assert self.generated, "model must be generated before compiling"
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
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

    def predict(self, dataset: datasets.Dataset, verbose=1, reverse_scaling=True):
        assert self.fitted, "model must be fitted before predicting"
        if reverse_scaling:
            return self.scaler.inverse_transform(self.model.predict(dataset.P_kzs, verbose=verbose))
        else:
            return self.model.predict(dataset.P_kzs, verbose=verbose)
    
    def evaluate(self, dataset: datasets.Dataset, verbose=1):
        assert self.fitted, "model must be fitted before evaluating"
        return self.model.evaluate(dataset.P_kzs, dataset.all_parameters, verbose=verbose)
                 

class BayConvPIPS(PIPSModel):
    def __init__(self, dataset: datasets.Dataset):
        super().__init__(dataset)
        self.kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /
                           tf.cast(self.trained_on, dtype=tf.float32))


    def generate(self, get_summary=True, n_conv=None, f_conv=None, kernel=None, \
                 n_dense=None, f_dense=None, activation=tf.nn.tanh, no_output=False, pool_size=None, \
                 normalize=True):
        if n_conv is not None:
            assert len(f_conv) == n_conv, "define independently number of filters per conv layer"
            assert len(kernel) == n_conv, "define independently kernel size for each conv layer"
            assert len(pool_size) == n_conv, "define independently pool size for each conv layer"
        if n_dense is not None:
            assert len(f_dense) == n_dense, "define independently number of filters per dense layer"
    
        if n_conv is None:
            y, x = self.map_shape
            if y < 10 and x < 10:
                n_conv = 2
                f_conv = [8, 16]
                kernel = [(2, 2), (2, 2)]
                pool_size = [(2, 2), (2, 2)]
            else:
                n_conv = 4
                f_conv = [4, 8, 16, 32]
                kernel = [(2, 2), (2, 2), (2, 2), (2, 2)]
                pool_size = [(2, 2), (2, 2), (2, 2), (2, 2)]

        if n_dense is None:
            n_dense = 3
            f_dense = [128, 64, 32]


        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape))
        if normalize:
            self.model.add(tf.keras.layers.Normalization())
        for i in range(n_conv):
            self.model.add(tfp.layers.Convolution2DFlipout(filters = f_conv[i], \
                        kernel_size=kernel[i], padding='same', activation=activation, \
                        kernel_divergence_fn=self.kl_divergence_function))
            self.model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size[i], padding="same"))

        self.model.add(tf.keras.layers.Flatten())

        for i in range(n_dense):
            self.model.add(tfp.layers.DenseFlipout(units = f_dense[i], \
                        activation=activation, \
                        kernel_divergence_fn=self.kl_divergence_function))
        
        if not no_output:
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
            self.optimizer = tf.keras.optimizers.Adam()
        else:
            self.optimizer = optimizer
        if loss is None:
            self.loss = lambda y_true, y_pred: -tf.reduce_mean(y_pred.log_prob(y_true))
        else:
            self.loss = loss
        if metrics is None:
            self.metrics = ["mse", "mae"]
        else:
            self.metrics = metrics

        super().compile()

class HeterogenousComponent(PIPSModel):
    pass

class HeterogenousBayConvPIPS(BayConvPIPS, HeterogenousComponent):
    def __init__(self, dataset: datasets.Dataset):
        super().__init__(dataset)
    
    def generate(self, get_summary=True, n_conv=None, f_conv=None, kernel=None, \
                 n_dense=0, f_dense=[], activation=tf.nn.tanh, no_output=True):
        super().generate(get_summary, n_conv, f_conv, kernel, \
                            n_dense, f_dense, activation, no_output)
        

class HeterogenousNetwork(PIPSModel):
    def __init__(self):
        self.components = []

    def add_component(self, component: HeterogenousComponent):
        assert component.generated, "component must be generated before adding"
        self.components.append(component)

    def generate(self, get_summary=True):
        assert len(self.components) > 0, "add at least one component before generating"
        for component in self.components:
            component.generate(get_summary=False)
        self.generated = True
        if get_summary:
            self.model.summary()

    def compile(self, optimizer=None, loss=None, metrics=None):
        assert self.generated, "model must be generated before compiling"
