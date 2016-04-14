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
        gen_data=None,
        num_iterations_before_checkpoint=100
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
        self.num_iterations_before_checkpoint = num_iterations_before_checkpoint

    def __call__(self, X):
        return self.predictor(X)

    def train(self, gen_data):
        if self.verbose:
            print("{:7}: {}".format("Epoch", "Avg. Training Loss"))
        avg_loss = 0
        for i in range(1, self.epochs + 1):
            X, y = gen_data(self.batch)
            loss = self.trainer(X, y)
            if self.verbose:
                avg_loss += (np.mean(loss) / self.num_iterations_before_checkpoint)
                if i % self.num_iterations_before_checkpoint == 0:
                    print("{:7d}: {}".format(i, np.mean(avg_loss)))
                    avg_loss = 0


    def fit(self, X, y, random_seed=901931823):
        if self.gen_data is None:
            _i = 0
            training_data = DataSet(X, y)
            rng = random.Random(random_seed)
            rng.shuffle(training_data.indices)
            num_training_instances = len(training_data.indices)
            def gen_data(n):
                nonlocal _i, training_data, num_training_instances
                _i %= num_training_instances
                d = num_training_instances - _i + 1
                if d < n:
                    list_of_indices = training_data.indices[_i:] + training_data.indices[:n-d]
                    rng.shuffle(training_data.indices)
                else:
                    list_of_indices = training_data.indices[_i:_i+n]
                _i += n
                _X = training_data.X.take(list_of_indices, axis=0)
                _y = training_data.y.take(list_of_indices, axis=0)
                return _X, _y
            self.train(gen_data)
        else:
            self.train(self.gen_data)
