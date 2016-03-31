import numpy as np
import theano
import theano.tensor as T
# Derived from code written by Michael Bowling


class LinearPredictor(object):
    @staticmethod
    def shared_weights(init_fn):
        return theano.shared(
            value=np.asarray(init_fn(), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )

    @staticmethod
    def xavier_initialized_weights(rng, n_in, n_out):
        return LinearPredictor.shared_weights(
            lambda: rng.uniform(
                low=-np.sqrt(6.0/(n_in)),
                high=np.sqrt(6.0/n_in),
                size=(n_in, n_out)
            )
        )

    @staticmethod
    def zero_weights(n_in, n_out):
        return LinearPredictor.shared_weights(lambda: np.zeros((n_in, n_out)))

    def __init__(self, rng, input_data, n_in, n_out, activation=T.tanh, init="xavier", W=None, b=None):
        if W is None:
            if init == "xavier":
                self.W = LinearPredictor.xavier_initialized_weights(rng, n_in, n_out)
            elif init == "zero":
                self.W = LinearPredictor.zero_weights(n_in, n_out)
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
        '''Matching loss for the identity activation function.'''
        return T.mean((self.output - y) ** 2, 0)

    def l1_loss(self, y):
        '''Convex loss for the identity activation function.'''
        return T.mean(abs(self.output - y), 0)

    def log_loss(self, y): # Assumes binary {0, 1} target y
        '''Matching loss for a sigmoid activation function.'''
        logprob = T.log(self.output)
        lognotprob = T.log(1.0-self.output)
        return -(T.sum(logprob * y) + T.sum(lognotprob * (1.0 - y)))/y.shape[0]

    def unnormalized_entropy_loss(self, y):
        '''Matching loss for an exponential activation function.'''
        return T.mean((y * T.log(y / self.output)) + self.output - y)


def exponential_activation(pre_image):
    '''
    Activation function that maps pre-images onto the space of positive
    real numbers
    '''
    return T.exp(pre_image)
