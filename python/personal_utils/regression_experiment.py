import numpy as np
import theano
import theano.tensor as T
try:
    from .sat_net import SatNet
except:
    from sat_net import SatNet


class DataSet(object):
    def __init__(self, X, y, indices):
        self.X = X
        self.y = y
        self.indices = indices


class Regressor(object):
    def __init__(self, X, y, learner, step_size=0.2, epochs=2000, batch=256):
        self.learner = learner
        self.trainer = theano.function(
            inputs=[X, y],
            outputs=self.learner.loss(y),
            updates=[
                (p, p - step_size * T.grad(T.mean(self.learner.loss(y)), p))
                for p in self.learner.params
            ]
        )
        self.predictor = theano.function(inputs=[X], outputs=self.learner.output)
        self.test = theano.function(
            inputs=[X, y],
            outputs=self.learner.loss(y)
        )
        self.epochs = epochs
        self.batch = batch

    def __call__(self, X):
        return self.predictor(X)

    def train(self, gen_data):
        print("{:7}: {}".format("Epoch", "Training Loss"))
        for i in range(self.epochs):
            loss = self.trainer(*gen_data(self.batch))
            if i % 100 == 0:
                print("{:7d}: {}".format(i, np.mean(loss)))


def run_experiment(training_data, testing_data, regression):
    def gen_data(n):
        list_of_indices = random.sample(training_data.indices, n)
        X = []
        y = []
        for i in list_of_indices:
            X.append(training_data.X[i])
            y.append(training_data.y[i])
        return X, y

    regression.train(gen_data)

    X_test = []
    y_test = []
    for i in testing_data.indices:
        X_test.append(testing_data.X[i])
        y_test.append(testing_data.y[i])
    return regression.test(X_test, y_test)


def construct_and_run_experiment(
    whole_data_set,
    regression,
    fraction_to_test=0.25
):
    first_testing_example_index = len(whole_data_set.indices) - (
        int(len(whole_data_set.indices) * fraction_to_test)
    )

    training_data = DataSet(
        whole_data_set.X,
        whole_data_set.y,
        whole_data_set.indices[:first_testing_example_index]
    )
    testing_data = DataSet(
        whole_data_set.X,
        whole_data_set.y,
        whole_data_set.indices[first_testing_example_index:]
    )

    testing_loss = run_experiment(
        training_data,
        testing_data,
        regression
    )
    print("\n### Testing Loss: {}".format(testing_loss))
    return regression


import random


def test_sat(step_size=0.2, epochs=2000, batch=256, nvars=64):
    def formula(x):
        return np.sum(x[0:2]) % 2

    def gen_data(n):
        X = np.random.randint(2, size=(n, nvars))
        y = np.asarray([[formula(x)] for x in X])
        return X, y

    all_X, all_y = gen_data(400000)
    indices = list(range(len(all_X)))
    random.shuffle(indices)
    input_dim = len(all_X[0])
    output_dim = len(all_y[0])

    rng = np.random

    X = T.matrix('X')
    y = T.matrix('y')

    predictor = construct_and_run_experiment(
        DataSet(all_X, all_y, indices),
        Regressor(
            X,
            y,
            SatNet(
                rng,
                X,
                input_dim,
                output_dim,
                (64,)
            ),
            step_size=step_size,
            epochs=epochs,
            batch=batch
        )
    )


if __name__ == "__main__":
    test_sat()
