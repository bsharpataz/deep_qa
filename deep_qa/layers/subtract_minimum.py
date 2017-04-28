from keras.layers import Layer
from keras import backend as K

class SubtractMinimum(Layer):
    '''
    This layer is used to normalize across a tensor axis.  Normalization is done by finding the minimum value across
    the specified axis, and then subtracting that value from all values (again, across the spcified axis).
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
            # Handle the case where the mask is one-dimension smaller than the inputs to mask all values for that
            # masked vector.
            if K.ndim(mask) == K.ndim(inputs) - 1:
                mask = K.expand_dims(mask)
                mask_tile_dimensions = [1] * K.ndim(inputs)
                mask_tile_dimensions[-1] = K.int_shape(inputs)[-1]
                mask = K.tile(mask, mask_tile_dimensions)
            # Make all masked values very large
            mask_flipped_and_scaled = K.cast(K.equal(mask, K.zeros_like(mask)), "float32") * 1000000.0
            dim_1_minimums = K.min(inputs + mask_flipped_and_scaled, axis=self.axis, keepdims=True)
        else:
            dim_1_minimums = K.min(inputs, axis=self.axis, keepdims=True)

        minimums = K.tile(dim_1_minimums, tile_dimensions)
        normalized = inputs - minimums
        return normalized
