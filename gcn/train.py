# -*- coding: utf-8 -*-

import argparse
import mlflow
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import yaml

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
                adjacency_matrices,
                sparse_input,
                sparse_nnz_values,
                dropout,
                activation,
                use_bias,
                use_gate,
                reg_parameter):
    model_input = tf.keras.Input(
        shape=(input_dim,),
        sparse=sparse_input
    )

    layer = model_input

    for lidx, fsize in enumerate(filter_sizes + [output_dim]):
        if dropout:
            if sparse_input and lidx == 0:
                layer = SparseDropout(
                    rate=dropout,
                    noise_shape=(sparse_nnz_values,)
                )(layer)
            else:
                layer = layers.Dropout(
                    rate=dropout
                )(layer)

        layer = GraphConvolution(
            units=fsize,
            supports=adjacency_matrices,
            input_shape=(input_dim,) if sparse_input and lidx == 0 else None,
            activation=activation if lidx <= len(filter_sizes) else None,
            use_bias=use_bias,
            use_gate=use_gate,
            kernel_regularizer=tf.keras.regularizers.l2(reg_parameter)
        )(layer)

    model = tf.keras.Model(inputs=[model_input], outputs=[layer])

    return model


def main(args):
    with open(args.configuration) as fh:
        config = yaml.load(fh, Loader=yaml.SafeLoader)

    np.random.seed(config.get("random_seed", 42))
    tf.random.set_seed(config.get("random_seed", 42))

    experiment_basename = os.path.basename(args.configuration).split(".yml")[0]
    search_type, experiment_date, experiment_hour, _ = experiment_basename.split("_")

    if args.early_stopping > 0:
        val_losses = []

    mlflow.set_experiment(
        "{}_{}_{}_{}".format(
            args.input_basename.split("/")[-1],
            search_type,
            experiment_date,
            experiment_hour
        )
    )
    with mlflow.start_run():
        mlflow.log_param("experiment_basename", experiment_basename)
        for param, value in config.items():
            mlflow.log_param(param, value)

        dataset_path = "{}.csv.gz".format(
            args.input_basename
        )

        graph_paths = []
        for edge_type in config.get("edge_types", ["5-gram"]):
            graph_paths.append("{}.{}.csv.gz".format(
                args.input_basename,
                edge_type
            ))

        if config.get("feature_type"):
            features_path = "{}.{}.mm".format(
                args.input_basename,
                config["feature_type"]
            )
        else:
            features_path = None

        adjs, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
            dataset_path=dataset_path,
            graph_paths=graph_paths,
            features_path=features_path,
            weighted_edges=config.get("weighted_edges", False)
        )

        model = build_model(
            input_dim=features.shape[1],
            output_dim=y_train.shape[1],
            filter_sizes=config.get("filter_sizes", [16]),
            adjacency_matrices=[preprocess_adj(adj) for adj in adjs],
            sparse_input=True,
            sparse_nnz_values=features.nnz,
            dropout=config.get("dropout", 0),
            activation=config.get("activation", "relu"),
            use_bias=config.get("use_bias", False),
            use_gate=config.get("use_gate", False),
            reg_parameter=config.get("reg_parameter", 0)
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("learning_rate", 0.1))
        loss = masked_softmax_cross_entropy
        metrics = [masked_accuracy, masked_f1_score]
        train_step = train_function()
        validation_step = evaluation_function()

        progress_bar = trange(config.get("epochs", 100))
        for epoch in progress_bar:
            train_results = train_step(features, y_train, model, train_mask,
                                       loss, optimizer, metrics)
            validation_results = validation_step(features, y_val, model,
                                                 val_mask, loss, metrics)

            tres = "loss: {:.3f} - acc: {:.3f} - f1: {:.3f}".format(*train_results)
            vres = "val_loss: {:.3f} - val_acc: {:.3f} - val_f1: {:.3f}".format(*validation_results)
            progress_bar.set_description("{} | {}".format(tres, vres))

            epoch_results = {
                "train_loss": train_results[0].numpy(),
                "train_accuracy": train_results[1].numpy(),
                "train_f1": train_results[2].numpy(),
                "validation_loss": validation_results[0].numpy(),
                "validation_accuracy": train_results[1].numpy(),
                "validation_f1": validation_results[2].numpy()
            }

            mlflow.log_metrics(epoch_results, step=epoch)

            if args.early_stopping > 0:
                val_losses.append(validation_results[0].numpy())
                if (len(val_losses) > args.early_stopping and
                        val_losses[-1] > np.mean(val_losses[-(args.early_stopping+1):-1])):
                    mlflow.log_param("early_stopping", True)
                    break

        if epoch >= config.get("epochs", 100) - 1:
            mlflow.log_param("early_stopping", False)

        if args.run_test:
            test_step = evaluation_function()
            test_results = test_step(features, y_test, model, test_mask, loss, metrics)

            mlflow.log_metrics({
                "test_loss": test_results[0].numpy(),
                "test_accuracy": test_results[1].numpy(),
                "test_f1": test_results[2].numpy()
            })

        final_predictions = pd.DataFrame(
            tf.nn.softmax(model(features)).numpy(),
            columns=labels
        )
        final_predictions["True"] = 0
        final_predictions["Prediction"] = final_predictions[labels].values.argmax(axis=1)
        final_predictions["Split"] = "Unlabel"
        final_predictions.loc[train_mask, "True"] = y_train[train_mask].argmax(axis=1)
        final_predictions.loc[train_mask, "Split"] = "Train"
        final_predictions.loc[val_mask, "True"] = y_val[val_mask].argmax(axis=1)
        final_predictions.loc[val_mask, "Split"] = "Validation"
        final_predictions.loc[test_mask, "True"] = y_test[test_mask].argmax(axis=1)
        final_predictions.loc[test_mask, "Split"] = "Test"

        predictions_file = "/tmp/{}_final_predictions.csv".format(experiment_basename)
        final_predictions.to_csv(
            predictions_file,
            index=False
        )
        mlflow.log_artifact(predictions_file)
        os.unlink(predictions_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_basename",
                        help="Basename of the dataset files to generate the inputs.")
    parser.add_argument("configuration",
                        help="Path to the json with the configuration for the experiment.")
    parser.add_argument("--early-stopping", default=0, type=int)
    parser.add_argument("--run-test", action="store_true")

    args = parser.parse_args()

    main(args)
