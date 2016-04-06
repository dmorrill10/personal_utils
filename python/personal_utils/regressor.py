import numpy as np
import theano
import theano.tensor as T


class DataSet(object):
    def __init__(self, X, y, indices):
        self.X = X
        self.y = y
        self.indices = indices


def rms_prop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    ''' From a theano tutorial on github '''
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def gradient_descent(cost, params, step_size=0.2):
    return [
        (p, p - step_size * T.grad(cost, p))
        for p in params
    ]


class Regressor(object):
    def __init__(self, X, y, learner, optimizer=lambda cost, params: gradient_descent(cost, params, step_size=0.2), epochs=2000, batch=256, verbose=False):
        self.learner = learner
        self.trainer = theano.function(
            inputs=[X, y],
            outputs=self.learner.training_loss(y),
            updates=optimizer(T.mean(self.learner.training_objective(y)), self.learner.params)
        )
        self.predictor = theano.function(inputs=[X], outputs=self.learner.output)
        self.test = theano.function(
            inputs=[X, y],
            outputs=self.learner.testing_loss(y)
        )
        self.epochs = epochs
        self.batch = batch
        self.verbose = verbose

    def __call__(self, X):
        return self.predictor(X)

    def train(self, gen_data):
        if self.verbose:
            print("{:7}: {}".format("Epoch", "Training Loss"))
        for i in range(self.epochs):
            loss = self.trainer(*gen_data(self.batch))
            if self.verbose and i % 100 == 0:
                print("{:7d}: {}".format(i, np.mean(loss)))
