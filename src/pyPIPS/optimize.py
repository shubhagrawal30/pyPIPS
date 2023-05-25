import optuna
import numpy as np
import tqdm

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.keras.backend.clear_session()

from pyPIPS.datasets import Dataset
from pyPIPS.models import BayConvPIPS

class BayConvPIPSOptimizer():
    def __init__(self, train: Dataset, val: Dataset, epochs=91, callback_epoch=15):
        self.train = train
        self.val = val
        self.study = None
        self.epochs = epochs
        self.n_conv = None
        self.f_conv = None
        self.kernel = None
        self.n_dense = None
        self.f_dense = None
        self.activation = tf.nn.tanh
        self.callback_epoch = callback_epoch

    def single_model_run(self, trial):
        # try:
        if 1:
            model = BayConvPIPS(self.train)
            if self.n_conv is None:
                self.n_conv = trial.suggest_int('n_conv', 2, 4)
            if self.f_conv is None:
                self.f_conv = [2 ** trial.suggest_int(f'log2(f_conv{i})', 2, 5) for i in range(self.n_conv)]
            if self.kernel is None:
                kernel = [trial.suggest_int(f'kernel{i}', 2, 5) for i in range(self.n_conv)]
                self.kernel = list(zip(kernel, kernel))
            if self.n_dense is None:
                self.n_dense = trial.suggest_int('n_dense', 2, 4)
            if self.f_dense is None:
                self.f_dense = [2 ** trial.suggest_int(f'log2(f_dense{i})', 5, 8) for i in range(self.n_dense)]

            model.generate(get_summary=False, n_conv=self.n_conv, f_conv=self.f_conv, kernel=self.kernel, \
                 n_dense=self.n_dense, f_dense=self.f_dense)
            model.compile()

            model.fit(epochs=self.epochs, verbose=0, callback_epoch=self.callback_epoch)
            return model.evaluate(self.val, verbose=0)[0]
        # except Exception as e:
        #     print(e)
        #     return np.inf

    def optimize(self, n_trials=20, n_conv=None, f_conv=None, kernel=None, \
                 n_dense=None, f_dense=None, activation=tf.nn.tanh):
        self.n_conv = n_conv
        self.f_conv = f_conv
        self.kernel = kernel
        self.n_dense = n_dense
        self.f_dense = f_dense
        self.activation = activation

        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.single_model_run, n_trials=n_trials)

    def get_best_model(self):
        if self.study is None:
            raise ValueError("No study has been run yet. Run optimize() first.")

        model = BayConvPIPS(self.train)
        if self.n_conv is None:
            self.n_conv = self.study.best_params['n_conv']
        if self.f_conv is None:
            self.f_conv = [2 ** self.study.best_params[f'log2(f_conv{i})'] for i in range(self.n_conv)]
        if self.kernel is None:
            kernel = [self.study.best_params[f'kernel{i}'] for i in range(self.n_conv)]
            self.kernel = list(zip(kernel, kernel))
        if self.n_dense is None:
            self.n_dense = self.study.best_params['n_dense']
        if self.f_dense is None:
            self.f_dense = [2 ** self.study.best_params[f'log2(f_dense{i})'] for i in range(self.n_dense)]

        model.generate(get_summary=False, n_conv=self.n_conv, f_conv=self.f_conv, kernel=self.kernel, \
                n_dense=self.n_dense, f_dense=self.f_dense)
        model.compile()

        model.fit(epochs=self.epochs, verbose=0, callback_epoch=self.callback_epoch)
        
        return model
        
    def get_study(self):
        return self.study
    
    def get_best_params(self):
        return self.study.best_params
    
    def get_best_value(self):
        return self.study.best_value
    
    def get_best_trial(self):
        return self.study.best_trial
