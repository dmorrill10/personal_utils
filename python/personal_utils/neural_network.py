import numpy as np
import theano
import theano.tensor as T
# Written by Michael Bowling


class NeuralNetworkLayer:
    @staticmethod
    def shared_weights(init_fn):
        return theano.shared(
            value=np.asarray(init_fn(), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )

    @staticmethod
    def xavier_initialized_weights(rng, n_in, n_out):
        return NeuralNetworkLayer.shared_weights(
            lambda: rng.uniform(
                low=-np.sqrt(6.0/(n_in)),
                high=np.sqrt(6.0/n_in),
                size=(n_in, n_out)
            )
        )

    @staticmethod
    def zero_weights(n_in, n_out):
        return NeuralNetworkLayer.shared_weights(lambda: np.zeros((n_in, n_out)))

    def __init__(self, rng, input_data, n_in, n_out, activation=T.tanh, init="xavier", W=None, b=None):
        if W is None:
            if init == "xavier":
                self.W = NeuralNetworkLayer.xavier_initialized_weights(rng, n_in, n_out)
            elif init == "zero":
                self.W = NeuralNetworkLayer.zero_weights(n_in, n_out)
            else:
                raise ValueError('Invalid init parameter value, "{}"'.format(init))
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b = b

        self.output = activation(T.dot(input_data, self.W) + self.b)
        self.params = (self.W, self.b)

    def l2_loss(self, y):
        return T.mean((self.output - y) ** 2, 0)

    def l1_loss(self, y):
        return T.mean(abs(self.output - y), 0)

    def log_loss(self, y): # Assumes binary {0, 1} target y
        logprob = T.log(self.output)
        lognotprob = T.log(1.0-self.output)
        return -(T.sum(logprob * y) + T.sum(lognotprob * (1.0 - y)))/y.shape[0]


class SatNet:
    def __init__(self, rng, input_data, n_in, n_out, layers):
        self.layers = []

        l_in = n_in
        for l_out in layers:
            layer = NeuralNetworkLayer(rng, input_data, l_in, l_out)
            self.layers.append(layer)
            input_data = layer.output
            l_in = l_out

        layer = NeuralNetworkLayer(rng, input_data, l_in, n_out, activation=T.nnet.sigmoid, init="zero")
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

    net = SatNet(rng, X, nvars, 1, (64,))

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
