import numpy as np
import theano
import theano.tensor as T
from personal_utils.learner import Learner
from personal_utils.linear_model import LinearModel
from personal_utils.regressor import Regressor, DataSet
from personal_utils.regression_experiment import construct_and_run_experiment
import random


def test_sat(epochs=2000, batch=256, step_size=0.2, nvars=64, rng_seed=390221039):
    np.random.seed(rng_seed)
    rng = np.random

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

    X = T.matrix('X')
    y = T.matrix('y')

    model_params = [
        (
            64,
            {
                'new_weights': lambda n_in, n_out: LinearModel.xavier_initialized_weights(
                    rng,
                    n_in,
                    n_out
                ),
                'activation': T.tanh
            }
        ),
        (
            1,
            {
                'new_weights': lambda n_in, n_out: LinearModel.zero_weights(
                    n_in,
                    n_out
                ),
                'activation': T.nnet.sigmoid
            }
        ),
    ]

    net = Learner(
        [m for m in LinearModel.every_model(X, nvars, model_params)],
        regularizer=lambda learner: 0
    )

    predictor, testing_loss = construct_and_run_experiment(
        DataSet(all_X, all_y, indices),
        Regressor(
            X,
            y,
            net,
            step_size=step_size,
            epochs=epochs,
            batch=batch,
            verbose=False
        )
    )
    # print("\n### Testing Loss: {}".format(testing_loss))
    assert testing_loss == np.array(0.0)
