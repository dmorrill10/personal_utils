import numpy as np
import theano
import theano.tensor as T
try:
    from .linear_predictor import LinearPredictor
except:
    from linear_predictor import LinearPredictor
# Originally written by Michael Bowling


class SatNet:
    def __init__(self, rng, input_data, n_in, n_out, *layers):
        self.layers = []

        l_in = n_in
        for l_out in layers:
            layer = LinearPredictor(rng, input_data, l_in, l_out)
            self.layers.append(layer)
            input_data = layer.output
            l_in = l_out

        layer = LinearPredictor(rng, input_data, l_in, n_out, activation=T.nnet.sigmoid, init="zero")
        self.layers.append(layer)

        self.input_data = input_data
        self.output = layer.output
        self.params = sum((l.params for l in self.layers), ())

    def loss(self, y):
        return self.layers[-1].log_loss(y)

    def accuracy(self, y): # Assumes binary {0, 1} targets
        return (T.sum((self.output > 0.5) * (y > 0.5)) + T.sum((self.output < 0.5) * (y < 0.5)))/y.shape[0]


def test_sat(epochs=2000, batch=256, stepsize=0.2, nvars=64):
    rng = np.random

    X = T.matrix('X')
    y = T.matrix('y')

    net = SatNet(rng, X, nvars, 1, 64)

    train = theano.function(inputs=[X,y],
                            outputs=net.loss(y),
                            updates=[(p, p - stepsize * T.grad(T.mean(net.loss(y)), p))
                                     for p in net.params])
    eval = theano.function(inputs=[X],
                           outputs=net.output)
    test = theano.function(inputs=[X,y],
                           outputs=[net.accuracy(y),net.loss(y)])

    def formula(x):
        return np.sum(x[0:2]) % 2

    def gen_data(n):
        X = np.random.randint(2, size=(n, nvars))
        y = np.asarray([[formula(x)] for x in X])
        return X, y

    print("{:7}: {}".format("Epoch", "Training Loss"))
    for i in range(epochs):
        loss = train(*gen_data(batch))
        print("{:7d}: {}".format(i, np.mean(loss)))

    print("TEST: {}".format(test(*gen_data(100000))))

if __name__ == "__main__":
    test_sat()
