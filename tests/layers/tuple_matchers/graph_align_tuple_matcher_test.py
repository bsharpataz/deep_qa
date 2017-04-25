# pylint: disable=no-self-use
import numpy as np
from numpy.testing import assert_array_almost_equal
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from scipy.stats import logistic

from deep_qa.layers.tuple_matchers.graph_align_tuple_matcher import GraphAlignTupleMatcher
from ...common.test_case import DeepQaTestCase


class TestWordOverlapTupleMatcher(DeepQaTestCase):
    def setUp(self):
        super(TestWordOverlapTupleMatcher, self).setUp()
        num_features = 5
        self.tuple1_input = Input(shape=(num_features,), dtype='float32', name="input1")
        self.tuple2_input = Input(shape=(num_features,), dtype='float32', name="input2")
        self.num_hidden_layers = 1
        self.hidden_layer_width = 2
        self.hidden_layer_activation = 'linear'
        self.match_layer = GraphAlignTupleMatcher(self.num_hidden_layers,
                                                   self.hidden_layer_width,
                                                   hidden_layer_activation=self.hidden_layer_activation)

        self.input_basic = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        self.input_has_zero = np.array([[0.0, 0.2, 0.3, 0.4, 0.5]])
        self.input_masked = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])


    def test_general_case(self):
        output = self.match_layer([self.tuple1_input, self.tuple2_input])
        model = Model([self.tuple1_input, self.tuple2_input], output)

        # Get the initial weights for use in testing
        dense_hidden_weights = K.eval(model.trainable_weights[0])
        score_weights = K.eval(model.trainable_weights[1])

        # Testing general unmasked case.
        # Features get fed into the inner NN.
        dense1_activation = np.dot(self.input_basic, dense_hidden_weights)
        final_score = np.dot(dense1_activation, score_weights)
        # Apply the final sigmoid activation function.
        desired_result = logistic.cdf(final_score)
        print(desired_result)
        result = model.predict([self.input_basic, self.input_basic])
        assert_array_almost_equal(result, desired_result)

    def test_masks_handled_correctly(self):
        # Test when one input has a zero, but isn't all padding.  Should not be masked
        not_all_masking = K.variable(self.input_has_zero)
        calculated_mask_exclude = K.eval(self.match_layer.compute_mask([not_all_masking, not_all_masking], [None, None]))
        assert calculated_mask_exclude.shape == (1, 1)
        assert_array_almost_equal(calculated_mask_exclude, np.array([[1]], dtype='int32'))

        # Test when all masking.
        all_masking = K.variable(self.input_masked)
        calculated_mask_include = K.eval(self.match_layer.compute_mask([all_masking, all_masking], [None, None]))
        assert calculated_mask_include.shape == (1, 1)
        assert_array_almost_equal(calculated_mask_include, np.array([[0]], dtype='int32'))
