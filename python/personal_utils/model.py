import numpy as np
import theano
from .losses_and_activations import identity_activation
# Derived from code written by Michael Bowling


class Model(object):
    @classmethod
    def shared_weights(self, init_fn, **kwargs):
        return theano.shared(
            value=np.asarray(init_fn(), dtype=theano.config.floatX),
            borrow=True,
            **kwargs
        )

    @classmethod
    def xavier_normalized_values(self, rng, n_in, n_out, **kwargs):
        '''Uniform random with particular bound inversely
           proportional to the number of input and output parameters
        '''
        bound = np.sqrt(6.0/(n_in + n_out))
        return rng.uniform(
            low=-bound,
            high=bound,
            **kwargs
        )

    @classmethod
    def xavier_initialized_weights(self, rng, n_in, n_out, **kwargs):
        '''Uniform random with particular bound inversely
           proportional to the number of input parameters
        '''
        bound = np.sqrt(6.0/n_in)
        return self.shared_weights(
            lambda: rng.uniform(
                low=-bound,
                high=bound,
                size=(n_in, n_out)
            ),
            **kwargs
        )

    @classmethod
    def every_model(self, input_data, n_in, model_params):
        l_in = n_in
        for l_out, kwargs in model_params:
            layer = self(input_data, l_in, l_out, **kwargs)
            yield layer
            input_data = layer.output
            l_in = l_out

    @classmethod
    def zero_weights(self, n_in, n_out, **kwargs):
        return self.shared_weights(lambda: np.zeros((n_in, n_out)), **kwargs)

    @classmethod
    def zero_bias(self, n_out, **kwargs):
        return theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            borrow=True,
            **kwargs
        )
