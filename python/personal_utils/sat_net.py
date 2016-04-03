import theano.tensor as T
try:
    from .linear_predictor import LinearPredictor
except:
    from linear_predictor import LinearPredictor


class SatNet:
    def __init__(
        self,
        rng,
        input_data,
        n_in,
        n_out,
        layers,
        final_activation=T.nnet.sigmoid,
        loss=lambda last_layer, y: last_layer.classification_log_loss(y)
    ):
        ''' Michael Bowling's SATNet implementation '''
        self.layers = []

        l_in = n_in
        for l_out in layers:
            layer = LinearPredictor(rng, input_data, l_in, l_out)
            self.layers.append(layer)
            input_data = layer.output
            l_in = l_out

        layer = LinearPredictor(
            rng,
            input_data,
            l_in,
            n_out,
            activation=final_activation,
            init="zero"
        )
        self.layers.append(layer)

        self.input_data = input_data
        self.output = layer.output
        self.params = sum((l.params for l in self.layers), ())
        self.loss = lambda y: loss(self.layers[-1], y)

    def accuracy(self, y): # Assumes binary {0, 1} targets
        return (
            T.sum((self.output > 0.5) * (y > 0.5)) +
            T.sum((self.output < 0.5) * (y < 0.5))
        ) / y.shape[0]
