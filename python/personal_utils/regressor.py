import numpy as np
import theano
import theano.tensor as T


class DataSet(object):
    def __init__(self, X, y, indices):
        self.X = X
        self.y = y
        self.indices = indices


class Regressor(object):
    def __init__(self, X, y, learner, step_size=0.2, epochs=2000, batch=256, verbose=False):
        self.learner = learner
        self.trainer = theano.function(
            inputs=[X, y],
            outputs=self.learner.training_loss(y),
            updates=[
                (p, p - step_size * T.grad(T.mean(self.learner.training_objective(y)), p))
                for p in self.learner.params
            ]
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
