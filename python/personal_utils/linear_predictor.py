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

    def __init__(
        self,
        rng,
        input_data,
        n_in,
        n_out,
        activation=T.tanh,
        training_loss=lambda last_layer, y: last_layer.cross_entropy_loss(y),
        testing_loss=lambda last_layer, y: last_layer.classification_loss(y),
        regularizer=lambda last_layer: last_layer.l2_regularizer(),
        regularization_param=1,
        init="xavier",
        W=None,
        b=None
    ):
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

        self.pre_image = T.dot(input_data, self.W) + self.b
        self.output = activation(self.pre_image)
        self.params = (self.W, self.b)
        self.regularization_param = regularization_param
        self.testing_loss = lambda y: testing_loss(self, y)
        self.training_loss = lambda y: training_loss(self, y) + regularizer(self)

    def l2_regularizer(self):
        return (self.regularization_param / 2.0) * T.dot(self.W.T, self.W)

    def l1_regularizer(self):
        return self.regularization_param * T.mean(abs(self.W), 0)

    def l2_loss(self, y):
        '''Matching loss for the identity activation function.'''
        return T.mean((self.output - y) ** 2, 0)

    def l1_loss(self, y):
        '''Convex loss for the identity activation function.'''
        return T.mean(abs(self.output - y), 0)

    def cross_entropy_loss(self, y):
        '''Matching loss for a sigmoid activation function.'''
        logprob = T.log(self.output)
        lognotprob = T.log(1.0-self.output)
        return -T.mean(
            (
                (logprob * y) +
                (lognotprob * (1.0 - y))
            ),
            0
        )

    def classification_loss(self, y):
        return (
            T.sum((self.output > 0.5) * (y > 0.5)) +
            T.sum((self.output < 0.5) * (y < 0.5))
        ) / y.shape[0]

    def unnormalized_entropy_loss(self, y):
        '''Matching loss for an exponential activation function.'''
        return T.mean(self.output - (y * T.log(self.output)))


def exponential_activation(pre_image):
    '''
    Activation function that maps pre-images onto the space of positive
    real numbers
    '''
    return T.exp(pre_image)

def identity_activation(pre_image):
    return pre_image
