"""Gumbel-Softmax layer implementation.
Reference: https://arxiv.org/pdf/1611.04051.pdf"""
from typing import Dict, List, Optional

# pylint: disable=E0401
from tensorflow import (Tensor, TensorShape, concat, one_hot, split, squeeze,
                        stop_gradient)
from tensorflow.keras.layers import Activation, Layer
from tensorflow.math import log
from tensorflow.nn import softmax
from tensorflow.random import categorical, uniform

TOL = 1e-20


def gumbel_noise(shape: TensorShape) -> Tensor:
    """Create a single sample from the standard (loc = 0, scale = 1) Gumbel distribution."""
    uniform_sample = uniform(shape, seed=0)
    return -log(-log(uniform_sample + TOL) + TOL)


class GumbelSoftmaxLayer(Layer):
    "A Gumbel-Softmax layer implementation that should be stacked on top of a categorical feature logits."

    def __init__(self, tau: float = 0.2, name: Optional[str] = None):
        super().__init__(name = name)
        self.tau = tau

    # pylint: disable=W0221, E1120
    def call(self, _input):
        """Computes Gumbel-Softmax for the logits output of a particular categorical feature."""
        noised_input = _input + gumbel_noise(_input.shape)
        soft_sample = softmax(noised_input/self.tau, -1)
        hard_sample = stop_gradient(squeeze(one_hot(categorical(log(soft_sample), 1), _input.shape[-1]), 1))
        return hard_sample, soft_sample


class ActivationInterface(Layer):
    """An interface layer connecting different parts of an incoming tensor to adequate activation functions.
    The tensor parts are qualified according to the passed processor object.
    Processed categorical features are sent to specific Gumbel-Softmax layers.
    Processed features of different kind are sent to a TanH activation.
    Finally all output parts are concatenated and returned in the same order.

    The parts of an incoming tensor are qualified by leveraging a data processor's in/out feature map.

    Example of how to get a col_map from a Data Processor ProcessorInfo attribute:
    >>> col_map = {k: [v.feat_names_in, v.feat_names_out] for k, v in ProcessorInfo._asdict().items() if v}"""

    def __init__(self, col_map: Dict[str, List[List[str]]], name: Optional[str] = None):
        """Arguments:
            col_map (Dict[str, List[List[str]]]): A map defining the processor pipelines input/output features.
            name (Optional[str]): Name of the layer"""
        super().__init__(name)

        self.cat_names_i, cat_names_o = col_map.get("categorical", [[],[]])
        num_names_i, num_names_o = col_map.get("numerical", [[],[]])

        self._cat_lens = None
        self._num_lens = None

        if self.cat_names_i:  # Get the length of each processed categorical feature's output block
            self._cat_lens = [len([col for col in cat_names_o \
            if ''.join(col.split('_')[:-1]) == cat_feat]) for cat_feat in self.cat_names_i]
        if num_names_i:  # Get the length of the numerical features output block
            self._num_lens = len(num_names_o)

    def call(self, _input):  # pylint: disable=W0221
        num_cols, cat_cols = split(_input, [self._num_lens if self._num_lens else 0, -1], 1, name='split_num_cats')
        cat_cols = split(cat_cols, self._cat_lens if self._cat_lens else 1, 1, name='split_cats')

        num_cols = [Activation('tanh', name='num_cols_activation')(num_cols)] if self._num_lens else []
        cat_cols = [GumbelSoftmaxLayer(name=name).call(col)[0] for name, col in zip(self.cat_names_i, cat_cols)] \
            if self._cat_lens else []
        return concat(num_cols+cat_cols, 1)
