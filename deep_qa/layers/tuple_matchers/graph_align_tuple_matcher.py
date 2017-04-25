from keras import backend as K
from keras import initializers, activations
from overrides import overrides

from ...tensors.backend import switch, apply_feed_forward
from ..masked_layer import MaskedLayer


class GraphAlignTupleMatcher(MaskedLayer):
    r"""
    This layer takes as input two tensors corresponding to a graph alignment and a dummy, and
    calculates entailment score of the alignment.

    Entailment is determined by passing a set of entailment features from the alignment into a
    shallow NN to get an entailment score.


    Inputs:

    - alignment_input, shape ``(batch size, num_features)``
      Any mask is ignored.

    - _ (the dummy input), shape ``(batch size, 1)``
      Unused.

    Output:

    - entailment score, shape ``(batch, 1)``

    Parameters
    ----------
    num_hidden_layers : int, default=1
        Number of hidden layers in the shallow NN.

    hidden_layer_width : int, default=4
        The number of nodes in each of the NN hidden layers.

    initialization : string, default='glorot_uniform'
        The initialization of the NN weights

    hidden_layer_activation : string, default='relu'
        The activation of the NN hidden layers

    final_activation : string, default='sigmoid'
        The activation of the NN output layer

    Notes
    _____
    This layer is incompatible with the ``WordsAndCharacters`` tokenizer.
    """

    def __init__(self, num_hidden_layers: int=1, hidden_layer_width: int=4,
                 initialization: str='glorot_uniform', hidden_layer_activation: str='tanh',
                 final_activation: str='sigmoid', **kwargs):
        # Parameters for the shallow neural network
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_layer_init = initialization
        self.hidden_layer_activation = hidden_layer_activation
        self.final_activation = final_activation
        self.hidden_layer_weights = []
        self.score_layer = None
        super(GraphAlignTupleMatcher, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(GraphAlignTupleMatcher, self).get_config()
        config = {'num_hidden_layers': self.num_hidden_layers,
                  'hidden_layer_width': self.hidden_layer_width,
                  'initialization': self.hidden_layer_init,
                  'hidden_layer_activation': self.hidden_layer_activation,
                  'final_activation': self.final_activation}
        config.update(base_config)
        return config

    def build(self, input_shape):
        super(GraphAlignTupleMatcher, self).build(input_shape)

        # Add the weights for the hidden layers.
        print("GATM (build) -- input_shape = ", input_shape)
        hidden_layer_input_dim = input_shape[0][1]
        print("GATM (build) -- hidden_layer_input_dim = ", hidden_layer_input_dim)
        for i in range(self.num_hidden_layers):
            hidden_layer = self.add_weight(shape=(hidden_layer_input_dim, self.hidden_layer_width),
                                           initializer=initializers.get(self.hidden_layer_init),
                                           name='%s_hiddenlayer_%d' % (self.name, i))
            self.hidden_layer_weights.append(hidden_layer)
            hidden_layer_input_dim = self.hidden_layer_width
        # Add the weights for the final layer.
        self.score_layer = self.add_weight(shape=(self.hidden_layer_width, 1),
                                           initializer=initializers.get(self.hidden_layer_init),
                                           name='%s_score' % self.name)

    def compute_output_shape(self, input_shapes):
        # pylint: disable=unused-argument
        return (input_shapes[0][0], 1)

    @overrides
    def compute_mask(self, input, input_mask=None):  # pylint: disable=unused-argument,redefined-builtin
        # Here, input_mask is ignored, because the input is plain word tokens. To determine the returned mask,
        # we want to see if either of the inputs is all padding (i.e. the mask would be all 0s), if so, then
        # the whole tuple_match should be masked, so we would return a 0, otherwise we return a 1.  As such,
        # the shape of the returned mask is (batch size, 1).
        input1 = input[0]
        mask = K.cast(K.any(input1, axis=1), 'uint8')
        return K.cast(K.expand_dims(mask), 'bool')
        # return K.cast(mask, 'bool')
        # return None

    def get_output_mask_shape_for(self, input_shape):  # pylint: disable=no-self-use
        # input_shape is [(batch_size, num_features), (batch_size, 1)]
        mask_shape = (input_shape[0][0], 1)
        return mask_shape

    @overrides
    def call(self, inputs, mask=None):
        # shape: (batch size, num_features)
        alignment_input, _ = inputs
        print("GATM -- K.int_shape(alignment_input): ", K.int_shape(alignment_input))

        raw_entailment = apply_feed_forward(alignment_input, self.hidden_layer_weights,
                                            activations.get(self.hidden_layer_activation))
        print("GATM -- K.int_shape(raw_entailment): ", K.int_shape(raw_entailment))
        # shape: (batch size, 1)
        final_score = activations.get(self.final_activation)(K.dot(raw_entailment, self.score_layer))
        print("GATM -- K.int_shape(final_score): ", K.int_shape(final_score))
        return final_score
