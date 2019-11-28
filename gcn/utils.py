# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import networkx as nx

from scipy import sparse as sps
from scipy.io import mmread


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_path, graph_path, features_path=None,
              weighted_edges=False, dtype=np.float32):
    """
    Load dataset and graph.

    Parameters
    ----------
    dataset_path : str
        Path to the csv file containing the data. It should have at least two
        columns: "Stance" which will be the target column, and "Split" that
        tells which row belongs to which dataset (train/test/validation/unlabel).
    graph_path : str
        Path to a csv file containing the adjacency matrix (only half of it,
        since it is symmetric) in coordinate format for the graph of the tweets.
    features_path : str
        If given, path to the feature representation of the nodes as a sparse
        Matrix Market matrix.
    weighted_edges : boolean
        Whether the graph has weighted edges or not.
    dtype : numpy type
        Type of the returned arrays
    """

    dataset = pd.read_csv(dataset_path)
    graph_data = pd.read_csv(graph_path)

    tweet_graph = nx.Graph()
    tweet_graph.add_weighted_edges_from(graph_data.values.tolist())

    adj = nx.adjacency_matrix(
        tweet_graph,
        weight="weight" if weighted_edges else None
    ).astype(dtype)
    adj.setdiag(0)

    if features_path:
        features = mmread(features_path)
    else:
        features = sps.eye(adj.shape[0])

    features = preprocess_features(features).tocsr().astype(dtype)

    labels = sorted(dataset["Stance"].unique())
    targets = pd.get_dummies(dataset["Stance"]).values.astype(dtype)

    idx_train = dataset.loc[dataset["Split"] == "Train"].index
    idx_val = dataset.loc[dataset["Split"] == "Validation"].index
    idx_test = dataset.loc[dataset["Split"] == "Test"].index

    train_mask = sample_mask(idx_train, targets.shape[0])
    val_mask = sample_mask(idx_val, targets.shape[0])
    test_mask = sample_mask(idx_test, targets.shape[0])

    y_shape = (features.shape[0], targets.shape[1])

    y_train = np.zeros(y_shape).astype(dtype)
    y_val = np.zeros(y_shape).astype(dtype)
    y_test = np.zeros(y_shape).astype(dtype)
    y_train[train_mask, :] = targets[train_mask, :]
    y_val[val_mask, :] = targets[val_mask, :]
    y_test[test_mask, :] = targets[test_mask, :]

    return (adj, features, labels, y_train, y_val, y_test,
            train_mask, val_mask, test_mask)


def preprocess_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sps.diags(r_inv)
    return r_mat_inv.dot(features)


def sparse_to_tuple(spmx):
    """Convert sparse matrix to tuple representation."""

    if not sps.isspmatrix_coo(spmx):
        spmx = spmx.tocoo()

    indices = np.vstack((spmx.row, spmx.col)).transpose()
    values = spmx.data
    shape = spmx.shape
    return indices, values, shape


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sps.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sps.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model"""
    return normalize_adj(adj + sps.eye(adj.shape[0])).astype(np.float32)
