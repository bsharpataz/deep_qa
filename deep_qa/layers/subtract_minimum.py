from keras import backend as K
from overrides import overrides

from deep_qa.layers.masked_layer import MaskedLayer

class SubtractMinimum(MaskedLayer):
    '''
    This layer is used to normalize across a tensor axis.  Normalization is done by finding the
    minimum value across the specified axis, and then subtracting that value from all values
    (again, across the spcified axis).  Note that this also works just fine if you want to find the
    minimum across more than one axis.

    Inputs:
        - A tensor with arbitrary dimension, and a mask of the same shape (currently doesn't
          support masks with other shapes).

    Output:
        - The same tensor, with the minimum across one (or more) of the dimensions subtracted.

    Parameters
    ----------
    axis: int
        The axis (or axes) across which to find the minimum.  Can be a single int, a list of ints,
        or None.  We just call `K.min` with this parameter, so anything that's valid there works
        here too.
    '''
    def __init__(self, axis: int, **kwargs):
        self.axis = axis
        super(SubtractMinimum, self).__init__(**kwargs)

    @overrides
    def compute_output_shape(self, input_shape): # pylint: disable=no-self-use
        return input_shape

    @overrides
    def compute_mask(self, inputs, mask=None):
        return mask

    @overrides
    def call(self, inputs, mask=None):
        if mask is not None:
            # Make all masked values very large.
            mask_flipped_and_scaled = K.cast(K.equal(mask, 0), "float32") * K.max(inputs)
            minimums = K.min(inputs + mask_flipped_and_scaled, axis=self.axis, keepdims=True)
        else:
            minimums = K.min(inputs, axis=self.axis, keepdims=True)
        normalized = inputs - minimums
        return normalized
