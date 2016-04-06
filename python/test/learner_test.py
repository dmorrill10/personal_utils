import numpy as np
import theano
import theano.tensor as T
from personal_utils.linear_model import LinearModel
from personal_utils.learner import Learner


def test_sat(epochs=2000, batch=256, stepsize=0.2, nvars=64, rng_seed=390221039):
    np.random.seed(rng_seed)
    rng = np.random

    X = T.matrix('X')
    y = T.matrix('y')

    model_params = [
        (
            64,
            {
                'new_weights': lambda n_in, n_out, **kwargs: LinearModel.xavier_initialized_weights(
                    rng,
                    n_in,
                    n_out,
                    **kwargs
                ),
                'activation': T.tanh
            }
        ),
        (
            1,
            {
                'new_weights': lambda n_in, n_out, **kwargs: LinearModel.zero_weights(
                    n_in,
                    n_out,
                    **kwargs
                ),
                'activation': T.nnet.sigmoid
            }
        ),
    ]

    net = Learner(
        [m for m in LinearModel.every_model(X, nvars, model_params)],
        regularizer=lambda learner: 0
    )

    train = theano.function(inputs=[X,y],
                          outputs=net.training_loss(y),
                          updates=[(p, p - stepsize * T.grad(T.mean(net.training_objective(y)), p))
                                   for p in net.params])
    eval = theano.function(inputs=[X],
                         outputs=net.output)
    test = theano.function(inputs=[X,y],
                         outputs=[net.testing_loss(y),net.training_loss(y)])

    def formula(x):
      return np.sum(x[0:2]) % 2

    def gen_data(n):
      X = np.random.randint(2, size=(n, nvars))
      y = np.asarray([[formula(x)] for x in X])
      return X, y

    # print("{:7}: {}".format("Epoch", "Training Loss"))
    for i in range(epochs):
      loss = train(*gen_data(batch))
      # print("{:7d}: {}".format(i, np.mean(loss)))

    test_results = test(*gen_data(100000))
    assert len(test_results) == 2
    assert test_results[0] == np.array(0.0)
    assert np.round(test_results[1], 3) == 0.006
