import numpy as np
import theano.tensor as T
# Derived from code written by Michael Bowling


def l2_loss(y_hat, y):
    '''Matching loss for the identity activation function.'''
    return T.mean((y_hat - y) ** 2, 0)


def l1_loss(y_hat, y):
    '''Convex loss for the identity activation function.'''
    return T.mean(abs(y_hat - y), 0)


def cross_entropy_loss(y_hat, y):
    '''Matching loss for a sigmoid activation function.'''
    logprob = np.log(y_hat)
    lognotprob = np.log(1.0-y_hat)
    return -(np.sum(logprob * y) + np.sum(lognotprob * (1.0 - y))) / y.shape[0]


def symbolic_cross_entropy_loss(y_hat, y):
    '''Matching loss for a sigmoid activation function.'''
    logprob = T.log(y_hat)
    lognotprob = T.log(1.0-y_hat)
    return -(T.sum(logprob * y) + T.sum(lognotprob * (1.0 - y))) / y.shape[0]


def unnormalized_entropy_loss(y_hat, y):
    '''Matching loss for an exponential activation function.'''
    return T.mean(y_hat - (y * T.log(y_hat)))


def classification_loss(y_hat, y):
    ''' Assumes binary {0, 1} targets '''
    return T.mean(T.neq((y_hat > 0.5), (y > 0.5)))


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
