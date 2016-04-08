import numpy as np
import theano
import theano.tensor as T
import random
from .regressor import Regressor, DataSet


def run_experiment(training_data, testing_data, regression):
    _i = 0
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
    return regression, testing_loss
