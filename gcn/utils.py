# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sps


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_path, graph_path, weighted_edges=False, dtype=np.float32):
    """
    Load dataset and graph.

    Parameters
    ----------
    dataset_path : str
        Path to the csv file containing the whole data. It should have at least two columns:
        "Stance" which will be the target column, and "Split" that tells which row belongs to which
        dataset (train/test/validation). This is the file obtained in the notebook
        `abortion_tweets_graph_construction.ipynb`.
    graph_path : str
        Path to a csv file containing the adjacency matrix (only half of it, since it is symmetric)
        in coordinate format for the graph of the tweets.
    weighted_edges : boolean
        Whether the graph has weighted edges or not.
    dtype : numpy type
        Type of the returned arrays
    """

    dataset = pd.read_csv(dataset_path)
    graph_data = pd.read_csv(graph_path)

    # This will eventually change to handle different feature engineering
    features = sps.eye(dataset.shape[0]).astype(dtype)

    tweet_graph = nx.Graph()
    tweet_graph.add_weighted_edges_from(graph_data.values.tolist())

    adj = nx.adjacency_matrix(
        tweet_graph,
        weight="weight" if weighted_edges else None
    ).astype(dtype)

    labels = pd.get_dummies(dataset["Stance"]).values.astype(dtype)

    idx_train = dataset.loc[dataset["Split"] == "Train"].index
    idx_val = dataset.loc[dataset["Split"] == "Validation"].index
    idx_test = dataset.loc[dataset["Split"] == "Test"].index

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros_like(labels)
    y_val = np.zeros_like(labels)
    y_test = np.zeros_like(labels)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


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
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    return normalize_adj(adj + sps.eye(adj.shape[0])).astype(np.float32)
