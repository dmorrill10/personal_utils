import numpy as np
import theano
import theano.tensor as T
from .log import log
import random


class DataSet(object):
    def __init__(self, X, y, indices=None):
        self.X = X
        self.y = y
        self.indices = indices if indices is not None else list(range(len(X)))


def rms_prop(cost, params, learning_rate=0.001, rho=0.9, epsilon=1e-6):
    ''' From a theano tutorial on github '''
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - learning_rate * g))
    return updates


def gradient_descent(cost, params, step_size=0.2):
    return [
        (p, p - step_size * T.grad(cost, p))
        for p in params
    ]


class Regressor(object):
    def __init__(
        self,
        X,
        y,
        learner,
        optimizer=lambda cost, params: gradient_descent(cost, params, step_size=0.2),
        epochs=2000,
        batch=256,
        verbose=False,
        gen_data=None
    ):
        self.learner = learner
        self.trainer = theano.function(
            inputs=[X, y],
            outputs=self.learner.training_loss(y),
            updates=optimizer(T.mean(self.learner.training_objective(y)), self.learner.params)
        )
        self.predictor = theano.function(inputs=[X], outputs=self.learner.output)
        self.test = lambda X, y: learner.testing_loss(self.predictor(X), y)
        self.epochs = epochs
        self.batch = batch
        self.verbose = verbose
        self.gen_data = gen_data

    def __call__(self, X):
        return self.predictor(X)

    def train(self, gen_data):
        if self.verbose:
            print("{:7}: {}".format("Epoch", "Training Loss"))
        for i in range(self.epochs):
            loss = self.trainer(*gen_data(self.batch))
            if self.verbose and i % 100 == 0:
                print("{:7d}: {}".format(i, np.mean(loss)))

    def fit(self, X, y):
        if self.gen_data is None:
            _i = 0
            training_data = DataSet(X, y)
            def gen_data(n):
                nonlocal _i, training_data
                _i %= len(training_data.indices)
                d = len(training_data.indices) - (_i + n)
                if d < n:
                    list_of_indices = training_data.indices[_i:] + training_data.indices[0:n-d]
                    random.shuffle(training_data.indices)
                else:
                    list_of_indices = training_data.indices[_i:_i+n]
                _i += n
                _X = training_data.X.take(list_of_indices, axis=0)
                _y = training_data.y.take(list_of_indices, axis=0)
                return _X, _y
            self.train(gen_data)
        else:
            self.train(self.gen_data)
