# -*- coding: utf-8 -*-

import scipy.sparse as sps
import tensorflow as tf

from tensorflow.keras import layers

from .utils import sparse_to_tuple


class GraphConvolution(layers.Layer):
    def __init__(self,
                 units,
                 supports,
                 input_shape=None,
                 activation=None,
                 use_bias=False,
                 use_gate=False,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.units = units
        self.supports = supports
        self._input_shape = input_shape
        self.has_sparse_input = input_shape is not None

        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.use_gate = use_gate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        self.A = []
        self.W = []

        if self.use_bias:
            self.b = []

        if self.use_gate:
            self.W_gate = []
            self.b_gate = []

        _input_shape = self._input_shape[-1] if self.has_sparse_input else input_shape[-1]

        for i in range(len(self.supports)):
            indices, values, shape = sparse_to_tuple(self.supports[i])
            self.A.append(tf.SparseTensor(
                indices=indices,
                values=values,
                dense_shape=shape
            ))

            self.W.append(self.add_weight(shape=(_input_shape, self.units),
                                          initializer=self.kernel_initializer,
                                          name="W_{}".format(i),
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True))

            if self.use_bias:
                self.b.append(self.add_weight(shape=(self.units,),
                                              initializer=self.bias_initializer,
                                              name="b",
                                              regularizer=self.bias_regularizer,
                                              constraint=self.bias_constraint,
                                              trainable=True))

            if self.use_gate:
                self.W_gate.append(self.add_weight(shape=(_input_shape, 1),
                                                   initializer="glorot_uniform",
                                                   name="W_gate_{}".format(i),
                                                   trainable=True))

                self.b_gate.append(self.add_weight(shape=(1,),
                                                   initializer="glorot_uniform",
                                                   name="b_gate",
                                                   trainable=True))

    def call(self, inputs):
        output = []
        for i in range(len(self.supports)):
            if isinstance(inputs, tf.SparseTensor):
                pre_sup = tf.sparse.sparse_dense_matmul(inputs, self.W[i])
                if self.use_gate:
                    pre_g = tf.sparse.sparse_dense_matmul(inputs, self.W_gate[i])
            elif sps.issparse(inputs):
                indices, values, shape = sparse_to_tuple(inputs)
                X = tf.SparseTensor(
                    indices=indices,
                    values=values,
                    dense_shape=shape
                )
                pre_sup = tf.sparse.sparse_dense_matmul(X, self.W[i])
                if self.use_gate:
                    pre_g = tf.sparse.sparse_dense_matmul(X, self.W_gate[i])
            else:
                pre_sup = tf.matmul(inputs, self.W[i])
                if self.use_gate:
                    pre_g = tf.matmul(inputs, self.W_gate[i])

            if self.use_bias:
                pre_sup += self.b[i]

            if self.use_gate:
                g = tf.nn.sigmoid(pre_g + self.b_gate[i])

                # This obligues the batch size to be equal to the number of
                # nodes in the adjacency matrix
                output.append(tf.sparse.sparse_dense_matmul(self.A[i], g * pre_sup))
            else:
                output.append(tf.sparse.sparse_dense_matmul(self.A[i], pre_sup))

        return self.activation(tf.math.add_n(output))

    def compute_output_shape(self, input_shape):
        batch_size = input_shape.get_shape().as_list()[0]
        return batch_size, self.units


class SparseDropout(layers.Layer):
    def __init__(self, rate, noise_shape, **kwargs):
        super(SparseDropout, self).__init__()

        self.rate = rate
        self.noise_shape = noise_shape

    def call(self, inputs, training=None):
        if not training:
            return inputs

        if isinstance(inputs, tf.SparseTensor):
            X = inputs
        elif sps.issparse(inputs):
            indices, values, shape = sparse_to_tuple(inputs)
            X = tf.SparseTensor(
                indices=indices,
                values=values,
                dense_shape=shape
            )
        else:
            raise TypeError("The input to the layer must be a sparse tensor")

        random_tensor = self.rate + tf.random.uniform(shape=self.noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)

        return tf.sparse.retain(X, dropout_mask) * (1. / (1. - self.rate))
