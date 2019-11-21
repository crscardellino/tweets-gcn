# -*- coding: utf-8 -*-

import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tqdm import trange

from .layers import GraphConvolution, SparseDropout
from .metrics import masked_accuracy, masked_f1_score, masked_softmax_cross_entropy
from .utils import load_data, preprocess_adj


def train_function():
    @tf.function
    def train_step(data, target, model, mask, loss, optimizer, metrics=None):
        with tf.GradientTape() as tape:
            logits = model(data, training=True)
            cost = loss(target, logits, mask)
        grads = tape.gradient(cost, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        results = [cost]

        if metrics is not None:
            for metric in metrics:
                results.append(metric(target, logits, mask))

        return results

    return train_step


def evaluation_function():
    @tf.function
    def evaluation_step(data, target, model, mask, loss, metrics=None):
        logits = model(data, training=False)
        cost = loss(target, logits, mask)

        results = [cost]

        if metrics is not None:
            for metric in metrics:
                results.append(metric(target, logits, mask))

        return results

    return evaluation_step


def build_model(input_dim,
                output_dim,
                filter_sizes,
                adjacency_matrix,
                sparse_input=True,
                dropout=None,
                activation="relu",
                use_bias=False,
                kernel_regularizer=None,
                kernel_initializer="glorot_uniform"):
    model_input = tf.keras.Input(
        shape=(input_dim),
        sparse=sparse_input
    )

    layer = model_input

    for lidx, fsize in enumerate(filter_sizes + [output_dim]):
        if dropout:
            if sparse_input and lidx == 0:
                layer = SparseDropout(
                    rate=dropout,
                    noise_shape=(input_dim,)
                )(layer)
            else:
                layer = layers.Dropout(
                    rate=dropout
                )(layer)

        layer = GraphConvolution(
            units=fsize,
            support=adjacency_matrix,
            input_shape=(input_dim,) if sparse_input and lidx == 0 else None,
            activation=activation if lidx <= len(filter_sizes) else None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )(layer)

    model = tf.keras.Model(inputs=[model_input], outputs=[layer])

    return model


def main(args):
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    # Load data
    adj, features, y_train, y_validation, y_test, train_mask, validation_mask, test_mask = load_data(
        args.dataset_path,
        args.graph_path,
        args.weighted_edges
    )

    model = build_model(
        input_dim=features.shape[1],
        output_dim=y_train.shape[1],
        filter_sizes=args.filter_sizes,
        adjacency_matrix=preprocess_adj(adj),
        sparse_input=True,
        dropout=args.dropout,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3)
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = masked_softmax_cross_entropy
    metrics = [masked_accuracy, masked_f1_score]
    train_step = train_function()
    validation_step = evaluation_function()

    progress_bar = trange(args.epochs)
    for _ in progress_bar:
        train_results = train_step(features, y_train, model, train_mask,
                                   loss, optimizer, metrics)
        validation_results = validation_step(features, y_validation, model,
                                             validation_mask, loss, metrics)

        tres = "loss: {:.3f} - acc: {:.3f} - f1: {:.3f}".format(*train_results)
        vres = "val_loss: {:.3f} - val_acc: {:.3f} - val_f1: {:.3f}".format(*validation_results)
        progress_bar.set_description("{} | {}".format(tres, vres))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("graph_path")
    parser.add_argument("--weighted-edges", action="store_true")
    parser.add_argument("--filter-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)

    args = parser.parse_args()

    main(args)
