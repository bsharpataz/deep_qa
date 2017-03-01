from keras import backend as K
from deep_qa.layers.wrappers.time_distributed import TimeDistributed

class TimeDistributedWithMask(TimeDistributed):
    """
    A ``TimeDistributed`` layer that passes mask computation on to the wrapped layer.

    Keras' ``TimeDistributed`` layer does not call the wrapped layer in ``compute_mask()``.  This
    handles the mask computation exactly as the computation of ``x`` in ``call()``: we reshape the
    input and mask, call ``compute_mask`` on the wrapped layer, then reshape the result to add back
    the timesteps.

    If you need to have a masking calculation done by the wrapped layer, you need to use this
    class.  However, in order to make this work we had to define a new method for the wrapped
    layer: ``get_output_mask_shape_for``.  In order to wrap a layer with this, you need to
    implement that method on the wrapped layer, which makes this not suitable to just be put in the
    base ``TimeDistributed`` class.
    """
    def compute_mask(self, x, input_mask=None):
        print("\nTimeDistributedWithMask\n")
        print("x =", x)
        print("input_mask =", input_mask)
        if not isinstance(input_mask, list):
            x = [x]
            input_mask = [input_mask]
        valid_masks = [m for m in input_mask if m is not None]
        if len(valid_masks) == 0:
            return None
        timesteps = K.int_shape(x[0])[1]
        print("timeseps:", timesteps)
        input_shape = [K.int_shape(x_i) for x_i in x]
        print("input_shape = ", input_shape)
        if len(x) == 1:
            input_shape = input_shape[0]
        reshaped_xs, reshaped_masks = self.reshape_inputs_and_masks(x, input_mask)
        print("reshaped_xs = ", reshaped_xs)
        print("reshaped_masks = ", reshaped_masks)
        output_mask = self.layer.compute_mask(reshaped_xs, input_mask=reshaped_masks)
        print("output_mask =", output_mask)
        print("output_mask shape =", K.int_shape(output_mask))
        if output_mask is None:
            return None

        input_to_get_output_mask_shape_for = [(input_shape_i[0],) + input_shape_i[2:] for input_shape_i in input_shape]
        if len(input_to_get_output_mask_shape_for) == 0:
            input_to_get_output_mask_shape_for = input_to_get_output_mask_shape_for[0]

        print("input_shape_... = ", input_to_get_output_mask_shape_for)

        output_mask_shape = self.layer.get_output_mask_shape_for(input_to_get_output_mask_shape_for)
        print("output_mask_shape =", output_mask_shape)
        reshaped_shape = (-1, timesteps) + output_mask_shape[1:]
        #reshaped_shape = (K.int_shape(x[0])[0], timesteps) + output_mask_shape[1:]
        print("reshaped_shape =", reshaped_shape)
        outputs = K.reshape(output_mask, reshaped_shape)
        return outputs
