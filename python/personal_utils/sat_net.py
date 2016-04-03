import theano.tensor as T
try:
    from .linear_predictor import LinearPredictor
except:
    from linear_predictor import LinearPredictor


class SatNet(object):
    def __init__(
        self,
        rng,
        input_data,
        n_in,
        n_out,
        layers,
        final_activation=T.nnet.sigmoid,
        training_loss=lambda last_layer, y: last_layer.cross_entropy_loss(y),
        testing_loss=lambda last_layer, y: last_layer.classification_loss(y),
        regularizer=lambda last_layer: last_layer.l2_regularizer(),
        regularization_param=1
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
        self.testing_loss = lambda y: testing_loss(self.layers[-1], y)
        self.training_loss = lambda y: training_loss(self.layers[-1], y) + regularizer(self.layers[-1])
