try:
    from .linear_model import LinearModel
    from .losses_and_activations import cross_entropy_loss, classification_loss
except ImportError:
    from linear_model import LinearModel
    from losses_and_activations import LinearModel
# Derived from code written by Michael Bowling


class Learner(object):
    def __init__(
        self,
        layers,
        training_loss=lambda learner, y: cross_entropy_loss(learner.output_model().output, y),
        testing_loss=lambda X, y: classification_loss(X, y),
        regularizer=lambda learner: learner.output_model().l2_regularizer(1)
    ):
        self.layers = layers
        self.input_data = self.input_model().input_data
        self.output = self.output_model().output
        self.params = sum((l.params for l in self.layers), ())
        self.complexity = sum(l.complexity for l in self.layers)
        self.testing_loss = testing_loss
        self.training_loss = lambda y: training_loss(self, y)
        self.training_objective = lambda y: self.training_loss(y) + regularizer(self)

    def input_model(self):
        return self.layers[0]

    def output_model(self):
        return self.layers[-1]
