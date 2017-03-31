from keras.layers import Layer
from keras import backend as K

class SubtractMinimum(Layer):
    '''

    '''
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(SubtractMinimum, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add the trainable weight variable for the noise parameter.
        super(SubtractMinimum, self).build(input_shape)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        axis_dimension = K.int_shape(inputs)[self.axis]
        tile_dimensions = [1] * K.ndim(inputs)
        tile_dimensions[self.axis] = axis_dimension

        if mask is not None:
            mask_flipped_and_scaled = K.cast(K.equal(mask, K.zeros_like(mask)), "float32") * 1000.0
            dim_1_minimums = K.min(inputs + mask_flipped_and_scaled, axis=self.axis, keepdims=True)
        else:
            dim_1_minimums = K.min(inputs, axis=self.axis, keepdims=True)

        minimums = K.tile(dim_1_minimums, tile_dimensions)
        normalized = inputs - minimums
        return normalized

    # def call(self, inputs, mask=None):
    #     axis_dimension = K.int_shape(inputs)[self.axis]
    #     tile_dimensions = [1] * K.ndim(inputs)
    #     tile_dimensions[self.axis] = axis_dimension
    #
    #     if mask is not None:
    #         mask_flipped_and_scaled = K.cast(K.equal(mask, K.zeros_like(mask)), "float32") * 1000.0
    #         zeros_in_inputs_scaled_up = K.cast(K.equal(inputs, K.zeros_like(inputs)), "float32") * 1000.0
    #         dim_1_minimums = K.min(inputs + zeros_in_inputs_scaled_up + mask_flipped_and_scaled, axis=self.axis, keepdims=True)
    #     else:
    #         zeros_in_inputs_scaled_up = K.cast(K.equal(inputs, K.zeros_like(inputs)), "float32") * 1000.0
    #         dim_1_minimums = K.min(inputs + zeros_in_inputs_scaled_up, axis=self.axis, keepdims=True)
    #
    #     minimums = K.tile(dim_1_minimums, tile_dimensions)
    #     normalized = inputs - minimums
    #     return normalized
