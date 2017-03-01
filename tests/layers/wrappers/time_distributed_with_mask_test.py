# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

import numpy
from keras.layers import Input
from deep_qa.layers.encoders import BOWEncoder
from deep_qa.layers.time_distributed_embedding import TimeDistributedEmbedding
from deep_qa.layers.wrappers.encoder_wrapper import EncoderWrapper
from deep_qa.layers.wrappers.time_distributed_with_mask import TimeDistributedWithMask
from deep_qa.layers.wrappers.output_mask import OutputMask
from deep_qa.layers.tuple_matchers.slot_similarity_tuple_matcher import SlotSimilarityTupleMatcher
from deep_qa.training.models import DeepQaModel

class TestTimeDistributedWithMask(TestCase):
    def test_handles_multiple_masks(self):
        background_input = Input(shape=(2, 3, 3), dtype='int32')
        background_input_2 = Input(shape=(2, 3, 3), dtype='int32')
        embedding = TimeDistributedEmbedding(input_dim=3, output_dim=5, mask_zero=True)
        embedded_background = embedding(background_input)
        embedded_background_2 = embedding(background_input_2)
        encoder = EncoderWrapper(EncoderWrapper(BOWEncoder(output_dim=2)))
        encoded_background = encoder(embedded_background)
        encoded_background_2 = encoder(embedded_background_2)
        time_distributed = TimeDistributedWithMask(SlotSimilarityTupleMatcher("cosine_similarity"))
        # Shape of input to OutputMask is       [(batch size, 2, 3, 5), (batch size, 2, 3, 5)]
        # Shape of input_mask to OutputMask is  [(batch size, 2, 3), (batch size, 2, 3)]
        # Expected output mask shape [(batch_size, 2, 3), (batch_size, 1, 3)]

        time_distributed_output = time_distributed([encoded_background, encoded_background_2])
        mask_output = OutputMask()(time_distributed_output)
        model = DeepQaModel(input=[background_input, background_input_2], output=mask_output)
        zeros = [0, 0, 0]
        non_zeros = [1, 1, 1]
        batch = numpy.asarray([[[zeros, zeros, non_zeros], [non_zeros, non_zeros, zeros]]]) # shape: (batch size, 2, 3, 3)
        # i.e., (1, 2, 3, 3)
        # gets encoded to shape: (batch size, 1, 2, 3, 5)
        batch_expected_mask = numpy.asarray([[[0, 0, 1], [1, 1, 0]]]) # shape: (batch size, 2, 3)
        # i.e., (1, 2, 3)
        actual_mask = model.predict([batch, batch])
        numpy.testing.assert_array_almost_equal([batch_expected_mask, batch_expected_mask], actual_mask)