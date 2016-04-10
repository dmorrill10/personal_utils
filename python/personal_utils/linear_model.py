import theano.tensor as T
from .model import Model
from .losses_and_activations import identity_activation
# Derived from code written by Michael Bowling


class LinearModel(Model):
    @classmethod
    def every_model(self, input_data, n_in, model_params):
        l_in = n_in
        for l_out, kwargs in model_params:
            layer = self(input_data, l_in, l_out, **kwargs)
            yield layer
            input_data = layer.output
            l_in = l_out

    def __init__(
        self,
        input_data,
        n_in,
        n_out,
        new_weights=None,
        new_bias=None,
        activation=identity_activation,
    ):
        if new_weights is None:
            new_weights = self.zero_weights
        if new_bias is None:
            new_bias = self.zero_bias
        self.input_data = input_data
        self.W = new_weights(n_in, n_out, name='W')
        self.b = new_bias(n_out, name='b')
        self.output = activation(T.dot(input_data, self.W) + self.b)
        self.params = (self.W, self.b)

    def l2_regularizer(self, param):
        return (param / 2.0) * T.dot(self.W.T, self.W)

    def l1_regularizer(self, param):
        return param * T.mean(abs(self.W), 0)
