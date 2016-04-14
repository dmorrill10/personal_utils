import numpy as np
import theano
import theano.tensor as T
# Derived from code written by Michael Bowling


def l2_loss(y_hat, y):
    '''Matching loss for the identity activation function.'''
    return T.mean((y_hat - y) ** 2, 0)


def compiled_l2_loss():
    '''Convex loss for the identity activation function.'''
    y = T.matrix('y')
    y_hat = T.matrix('y_hat')
    return theano.function(inputs=[y_hat, y], outputs=l2_loss(y_hat, y))


def l1_loss(y_hat, y):
    '''Convex loss for the identity activation function.'''
    return T.mean(abs(y_hat - y), 0)


def compiled_l1_loss():
    '''Convex loss for the identity activation function.'''
    y = T.matrix('y')
    y_hat = T.matrix('y_hat')
    return theano.function(inputs=[y_hat, y], outputs=l1_loss(y_hat, y))


def symbolic_cross_entropy_loss(y_hat, y):
    '''Matching loss for a sigmoid activation function.'''
    logprob = T.log(y_hat)
    lognotprob = T.log(1.0-y_hat)
    return -(T.sum(logprob * y) + T.sum(lognotprob * (1.0 - y))) / y.shape[0]


def compiled_cross_entropy_loss():
    '''Matching loss for a sigmoid activation function.'''
    y = T.matrix('y')
    y_hat = T.matrix('y_hat')
    return theano.function(inputs=[y_hat, y], outputs=symbolic_cross_entropy_loss(y_hat, y))


def unnormalized_entropy_loss(y_hat, y):
    '''Matching loss for an exponential activation function.'''
    return T.mean(y_hat - (y * T.log(y_hat)))


def l1_loss_for_exponential_predictions(y_hat, y):
    ''' Not sure if this is convex when y_hat is an exponential function
        of a pre-image, but it makes sense to me.
    '''
    return T.mean(abs(T.log(y + 1e-10) - T.log(y_hat)))


def symbolic_classification_loss(y_hat, y):
    ''' Assumes binary {0, 1} targets '''
    return T.mean(T.neq((y_hat > 0.5), (y > 0.5)))


def compiled_classification_loss():
    ''' Assumes binary {0, 1} targets '''
    y = T.matrix('y')
    y_hat = T.matrix('y_hat')
    return theano.function(inputs=[y_hat, y], outputs=symbolic_classification_loss(y_hat, y))


def positive_vs_negative_cross_entropy_loss(y_hat, y):
    '''Matching loss for a sigmoid activation function.'''
    return cross_entropy_loss(y_hat, y > 0.0)


def positive_vs_negative_classification_loss(y_hat, y):
    return T.mean(T.neq((y_hat > 0.5), (y > 0.0)))


def exponential_activation(pre_image):
    '''
    Activation function that maps pre-images onto the space of positive
    real numbers
    '''
    return T.exp(pre_image)


def identity_activation(pre_image):
    return pre_image
