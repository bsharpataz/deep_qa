# pylint: disable=no-self-use
import numpy as np
from numpy.testing import assert_array_almost_equal

from keras.layers import Input, Masking
from keras.models import Model
from deep_qa.layers.subtract_minimum import SubtractMinimum
from ..common.test_case import DeepQaTestCase


class TestSubtractMinimum(DeepQaTestCase):
    def test_general_case(self):

        input_layer = Input(shape=(4,3,), dtype='float32', name="input")
        subtract_minimum_layer = SubtractMinimum(axis=1)
        normalized_input = subtract_minimum_layer(input_layer)

        model = Model([input_layer], normalized_input)
        # Testing general unmasked 1D case.
        unnormalized_tensor = np.array([[[0.1, 0.1, 0.1],
                                         [0.2, 0.3, 0.4],
                                         [0.5, 0.4, 0.6],
                                         [0.5, 0.4, 0.6]]])
        result = model.predict([unnormalized_tensor])

        assert_array_almost_equal(result, np.array([[[0.0, 0.0, 0.0],
                                                     [0.1, 0.2, 0.3],
                                                     [0.4, 0.3, 0.5],
                                                     [0.4, 0.3, 0.5]]]))

        # Testing masked batched case.
        # By setting the mast value to 0.1. should ignore this value when deciding the minimum
        mask_layer = Masking(mask_value=0.1)
        masked_input = mask_layer(input_layer)
        normalized_masked_input = subtract_minimum_layer(masked_input)
        masking_model = Model([input_layer], normalized_masked_input)

        masked_result = masking_model.predict([unnormalized_tensor])

        # Note the masked first row gets passed to the layer as all 0.0s, hence the desired values.
        assert_array_almost_equal(masked_result, np.array([[[-0.2, -0.3, -0.4],
                                                            [0.0, 0.0, 0.0],
                                                            [0.3, 0.1, 0.2],
                                                            [0.3, 0.1, 0.2]]]))
