:mod:`lasagne.layers`
=====================

.. automodule:: lasagne.layers


Helper functions
----------------

.. autofunction:: get_output
.. autofunction:: get_output_shape
.. autofunction:: get_all_layers
.. autofunction:: get_all_params
.. autofunction:: get_all_bias_params
.. autofunction:: get_all_non_bias_params
.. autofunction:: count_params
.. autofunction:: get_all_param_values
.. autofunction:: set_all_param_values


Layer base classes
------------------

.. autoclass:: Layer
   :members:

.. autoclass:: MergeLayer
    :members:

Layer classes: network input
----------------------------

.. autoclass:: InputLayer
   :members:

Layer classes: dense layers
---------------------------

.. autoclass:: DenseLayer
   :members:

.. autoclass:: NonlinearityLayer
   :members:

.. autoclass:: NINLayer
    :members:

Layer classes: convolutional layers
-----------------------------------

.. autoclass:: Conv1DLayer
    :members:

.. autoclass:: Conv2DLayer
    :members:

Layer classes: pooling layers
-----------------------------

.. autoclass:: MaxPool1DLayer
    :members:

.. autoclass:: Pool2DLayer
    :members:

.. autoclass:: MaxPool2DLayer
    :members:

.. autoclass:: GlobalPoolLayer
    :members:

.. autoclass:: FeaturePoolLayer
    :members:

.. autoclass:: FeatureWTALayer
    :members:

Layer classes: noise layers
---------------------------

.. autoclass:: DropoutLayer
    :members:

.. autoclass:: dropout

.. autoclass:: GaussianNoiseLayer
    :members:

Layer classes: shape layers
---------------------------

.. autoclass:: ReshapeLayer
    :members:

.. autoclass:: reshape

.. autoclass:: FlattenLayer
    :members:

.. autoclass:: flatten

.. autoclass:: DimshuffleLayer
    :members:

.. autoclass:: dimshuffle

.. autoclass:: PadLayer
    :members:

.. autoclass:: pad

.. autoclass:: SliceLayer


Layer classes: merge layers
---------------------------

.. autoclass:: ConcatLayer
    :members:

.. autoclass:: concat

.. autoclass:: ElemwiseMergeLayer
    :members:

.. autoclass:: ElemwiseSumLayer
    :members:

Layer classes: embedding layers
-------------------------------

.. autoclass:: EmbeddingLayer
    :members:

Layer classes: recurrent layers
-------------------------------

.. automodule:: lasagne.layers.recurrent

.. autoclass:: CustomRecurrentLayer
    :members:

.. autoclass:: RecurrentLayer
    :members:

.. autoclass:: LSTMLayer
    :members:

.. autoclass:: GRULayer
    :members:

.. autoclass:: Gate
    :members:

:mod:`lasagne.layers.corrmm`
============================

.. automodule:: lasagne.layers.corrmm
    :members:


:mod:`lasagne.layers.cuda_convnet`
==================================

.. automodule:: lasagne.layers.cuda_convnet
    :members:


:mod:`lasagne.layers.dnn`
=========================

.. automodule:: lasagne.layers.dnn
    :members:
