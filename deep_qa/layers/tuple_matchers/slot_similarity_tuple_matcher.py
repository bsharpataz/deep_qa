from keras import backend as K
from keras.layers import Layer
from keras import initializations, activations
from overrides import overrides

from ...tensors.backend import apply_feed_forward
from ...tensors.similarity_functions import similarity_functions


class SlotSimilarityTupleMatcher(Layer):
    """
    Like other ``TupleMatch`` layers, this layer takes as input two tensors corresponding to two tuples,
    an answer tuple and a background tuple, and calculates the degree to which the background tuple
    `entails` the answer tuple.  In this layer, each input slot is represented as a dense embedding, so
    to determine entailment we find the cosine similarity between these embedded slot representations,
    i.e., the similarity between the first slot in each, then the second slot in each, etc.
    This generates a set of similarity features equal to the number of slots in the tuples, which are
    then fed to a shallow NN with output of size one.  The output of this NN is considered to be the
    entailment score for the two tuples.

    Inputs:
        - tuple_1_input (the answer tuple), shape ``(batch size, num_slots, embed_dimensions)``.  There also
          needs to be a corresponding mask of shape (batch size, num_slots) (or None) that indicates whether
          a given slot was all padding.

        - tuple_2_input (the background_tuple), shape ``(batch size, num_slots, embed_dimensions)``,
          and again, there needs to be a corresponding mask of shape (batch size, num_slots) (or None)
          that indicates whether a given slot was all padding.

    Output:
        - entailment score, shape ``(batch, 1)``

    Parameters
    ----------
    - similarity_function_choice: str
        The similarity function used to compare the slots of the inputs.

    - num_hidden_layers : int, default=1
        Number of hidden layers in the shallow NN.

    - hidden_layer_width : int, default=4
        The number of nodes in each of the NN hidden layers.

    - initialization : string, default='glorot_uniform'
        The initialization of the NN weights

    - hidden_layer_activation : string, default='relu'
        The activation of the NN hidden layers

    - final_activation : string, default='sigmoid'
        The activation of the NN output layer

    Notes
    _____
    This layer is incompatible with the WordsAndCharacters tokenizer.
    """
    def __init__(self, similarity_function_choice, num_hidden_layers: int=1, hidden_layer_width: int=4,
                 initialization: str='glorot_uniform', hidden_layer_activation: str='tanh',
                 final_activation: str='sigmoid', **kwargs):
        self.input_dim = None
        self.supports_masking = True
        self.similarity_function_choice = similarity_function_choice
        self.similarity_function = similarity_functions[self.similarity_function_choice](name="slot_similarity")
        # Parameters for the shallow neural network
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_layer_init = initialization
        self.hidden_layer_activation = hidden_layer_activation
        self.final_activation = final_activation
        self.hidden_layer_weights = []
        self.score_layer = None
        super(SlotSimilarityTupleMatcher, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(SlotSimilarityTupleMatcher, self).get_config()
        config = {'similarity_function_choice': self.similarity_function_choice,
                  'num_hidden_layers': self.num_hidden_layers,
                  'hidden_layer_width': self.hidden_layer_width,
                  'initialization': self.hidden_layer_init,
                  'hidden_layer_activation': self.hidden_layer_activation,
                  'final_activation': self.final_activation}
        config.update(base_config)
        return config

    def get_output_shape_for(self, input_shapes):
        # pylint: disable=unused-argument
        return (input_shapes[0][0], 1)

    def build(self, input_shape):
        super(SlotSimilarityTupleMatcher, self).build(input_shape)

        # Add the weights for the hidden layers.
        hidden_layer_input_dim = input_shape[0][1]
        for i in range(self.num_hidden_layers):
            hidden_layer = self.add_weight(shape=(hidden_layer_input_dim, self.hidden_layer_width),
                                           initializer=initializations.get(self.hidden_layer_init),
                                           name='%s_hiddenlayer_%d' % (self.name, i))
            self.hidden_layer_weights.append(hidden_layer)
            hidden_layer_input_dim = self.hidden_layer_width
        # Add the weights for the final layer.
        self.score_layer = self.add_weight(shape=(self.hidden_layer_width, 1),
                                           initializer=initializations.get(self.hidden_layer_init),
                                           name='%s_score' % self.name)

    @overrides
    def compute_mask(self, input, input_mask=None):  # pylint: disable=unused-argument,redefined-builtin
        # Here, we want to see if either of the inputs is all padding (i.e. the mask would be all 0s).
        # If so, then the whole tuple_match should be masked, so we would return a 0, otherwise we
        # return a 1.  As such, the shape of the returned mask is (batch size, 1).
        if input_mask is None or input_mask == [None, None]:
            return None
        # Each of the two masks in input_mask are of shape: (batch size, num_slots)
        mask1, mask2 = input_mask
        mask1 = K.cast(K.any(mask1), 'int32')
        mask2 = K.cast(K.any(mask2), 'int32')
        return (mask1 + mask2) >= 2

    def call(self, x, mask=None):
        tuple1_input, tuple2_input = x      # tuple1 shape: (batch size, num_slots, embed_dimension)
                                            # tuple2 shape: (batch size, num_slots, embed_dimension)
        # Check that the tuples have the same number of slots.
        assert K.int_shape(tuple1_input)[1] == K.int_shape(tuple2_input)[1]

        # Calculate the cosine similarities.
        # shape: (batch size, num_slots)
        similarities = self.similarity_function.compute_similarity(tuple1_input, tuple2_input)

        # Remove any similarities if one of the corresponding slots was all padding.
        # TODO(becky): need this to take a mask for each input of None or shape: (batch size, num_slots)
        tuple1_mask, tuple2_mask = mask
        # Make a masked version of similarities which remomves similarities from slots which were all
        # padding in either tuple.
        # shape: (batch size, num_slots)
        masked_similarities = similarities
        if tuple1_mask is not None:
            masked_similarities *= K.cast(tuple1_mask, "float32")
        if tuple2_mask is not None:
            masked_similarities *= K.cast(tuple2_mask, "float32")

        # shape: (batch size, hidden_layer_width)
        raw_entailment = apply_feed_forward(masked_similarities, self.hidden_layer_weights,
                                            activations.get(self.hidden_layer_activation))
        # shape: (batch size, 1)
        final_score = activations.get(self.final_activation)(K.dot(raw_entailment, self.score_layer))

        return final_score
