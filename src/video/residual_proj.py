from linear import Linear


class ResidualProj(Linear):
    """Linear whose output is added to the residual stream; gets 1/sqrt(2*n_layer) init.

    Purely a marker -- forward is Linear's. GPT._init_weights looks for the type.
    """
