#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import re
import string
import sys
import unidecode

from collections import defaultdict
from gensim import corpora, models
from itertools import chain
from joblib import Parallel, delayed
from nltk import ngrams
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import TweetTokenizer
from operator import itemgetter
from scipy import sparse as sps


def normalize_token(token, **kwargs):
    if kwargs.get("remove_hashtags") and token.startswith("#"):
        return ""

    if kwargs.get("remove_links") and token.startswith("http"):
        return ""

    if kwargs.get("remove_mentions") and token.startswith("@"):
        return ""

    if kwargs.get("remove_numeric") and token.isnumeric():
        return ""

    if kwargs.get("split_hashtags") and token.startswith("#"):
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token[1:])
        token = "#" + ":".join([m.group(0) for m in matches])
    elif kwargs.get("normalize_hashtags") and token.startswith("#"):
        token = "<hashtag>"

    if kwargs.get("normalize_mentions") and token.startswith("@"):
        token = "<user>"

    if kwargs.get("tweet_lowercase"):
        token = token.lower()

    return unidecode.unidecode(token)


def split_hashtags(token):
    if token.startswith("#"):
        return token[1:].split(":")
    else:
        return [token]


def normalize_tweet(tweet, stopwords=set(), punctuation=set(), **kwargs):
    tweet = [normalize_token(t, **kwargs).strip() for t in tweet
             if t not in stopwords and t not in punctuation]

    if kwargs.get("split_hashtags"):
        tweet = list(chain(*map(split_hashtags, tweet)))

    return [t for t in tweet if t != ""]


def extract_hashtags(tokens, hashtag_ignore=set()):
    return sorted(set([
        t for t in tokens
        if t.startswith("#") and
        t.strip() != "#" and
        t.lower()[1:] not in hashtag_ignore
    ]))


def extract_mentions(tokens, mentions_ignore=set()):
    return sorted(set([
        t for t in tokens
        if t.startswith("@") and
        t.strip() != "@" and
        t.lower()[1:] not in mentions_ignore
    ]))


def extract_ngrams(tokens, n=3):
    return sorted(set([
        "_".join(ngram) for ngram in ngrams(tokens, n=n)
    ]))


def extract_toptfidf(tfidf_tweet, k=5):
    return [
        t[0] for t in sorted(tfidf_tweet, key=itemgetter(1), reverse=True)[:k]
    ]


def build_adjacency_matrix(graph_type, data):
    adjacency = []
    for idx, row_i in data.iterrows():
        # Needed for NetworkX to keep track of all existing nodes
        # (even isolated ones)
        adjacency.append((row_i["ID"], row_i["ID"], 0))
        # Only store a triangular matrix (the matrix is symmetric)
        for _, row_j in data.loc[idx+1:].iterrows():
            # TODO: Is this the best way to weight edges?
            edge_weight = len(
                set(row_i[graph_type]).intersection(row_j[graph_type])
            )
            if edge_weight > 0:
                adjacency.append((row_i["ID"], row_j["ID"], edge_weight))
    return graph_type, adjacency


def main(args):
    print("Loading data", file=sys.stderr)
    dataset = pd.read_csv(args.dataset_input)

    if args.supervised_only:
        print("Filtering unsupervised", file=sys.stderr)
        dataset = dataset[dataset["Stance"] != "UNK"]

    print("Tokenizing tweets", file=sys.stderr)
    tweet_tokenizer = TweetTokenizer(
        reduce_len=args.reduce_tweet_word_len
    )
    dataset["TokenizedTweet"] = dataset["Tweet"].apply(
        tweet_tokenizer.tokenize
    )

    print("Normalizing tweets", file=sys.stderr)
    punctuation_symbols = set(string.punctuation) \
        if args.remove_punctuation else set()
    stopwords = set(nltk_stopwords.words("english")) \
        if args.remove_stopwords else set()
    dataset["NormalizedTweet"] = dataset["TokenizedTweet"].apply(
        lambda t: normalize_tweet(
            tweet=t,
            stopwords=stopwords,
            punctuation=punctuation_symbols,
            remove_hashtags=args.remove_hashtags,
            remove_links=args.remove_links,
            remove_mentions=args.remove_mentions,
            remove_numeric=args.remove_numeric,
            normalize_hashtags=args.normalize_hashtags,
            normalize_mentions=args.normalize_mentions,
            split_hashtags=args.split_hashtags,
            tweet_lowercase=args.tweet_lowercase
        )
    )

    print("Building vocabulary", file=sys.stderr)
    tweets_vocab = corpora.Dictionary(dataset["NormalizedTweet"])
    tweets_vocab.filter_extremes(
        no_below=args.min_docs,
        no_above=args.max_docs
    )

    print("Building bag-of-words features", file=sys.stderr)
    bow_corpus = dataset["NormalizedTweet"].apply(
        tweets_vocab.doc2bow
    ).tolist()
    corpora.MmCorpus.serialize(
        "{}.bow.mm".format(args.output_basename),
        bow_corpus
    )

    print("Building TF-IDF features", file=sys.stderr)
    tfidf_model = models.TfidfModel(
        bow_corpus,
        dictionary=tweets_vocab
    )
    tfidf_corpus = tfidf_model[bow_corpus]
    corpora.MmCorpus.serialize(
        "{}.tfidf.mm".format(args.output_basename),
        tfidf_corpus
    )

    print("Extracting graph information", file=sys.stderr)
    graph_types = []

    if args.graph_hashtags:
        dataset["hashtags"] = dataset["TokenizedTweet"].apply(
            lambda t: extract_hashtags(
                t,
                set(map(lambda t: t.lower(), args.ignore_hashtags))
            )
        )
        graph_types.append("hashtags")

    if args.graph_mentions:
        dataset["mentions"] = dataset["TokenizedTweet"].apply(
            lambda t: extract_mentions(
                t,
                set(map(lambda t: t.lower(), args.ignore_mentions))
            )
        )
        graph_types.append("mentions")

    for n in args.graph_ngrams:
        dataset["{}-gram".format(n)] = dataset["TokenizedTweet"].apply(
            lambda t: extract_ngrams(t, n)
        )
        graph_types.append("{}-gram".format(n))

    for k in args.graph_tfidf:
        dataset["top-{}-tfidf".format(k)] = dataset["ID"].apply(
            lambda idx: extract_toptfidf(tfidf_corpus[idx], k)
        )
        graph_types.append("top-{}-tfidf".format(k))

    print("Building graphs", file=sys.stderr)
    adjacencies = dict(
        Parallel(n_jobs=-1, verbose=10)(
            delayed(build_adjacency_matrix)(
                graph_type, dataset.loc[:, ["ID", graph_type]]
            ) for graph_type in graph_types
        )
    )

    if args.graph_document_word:
        print("Building document_word_graph", file=sys.stderr)
        tweets_corpus = dataset["NormalizedTweet"].apply(
            tweets_vocab.doc2idx
        ).tolist()

        # Word-Word Co-occurrence Matrix
        word_word_count = defaultdict(int)
        window_size = args.graph_document_word_window
        for tweet in tweets_corpus:
            for idx, ctoken in enumerate(tweet):
                if ctoken == -1:
                    continue
                for wtoken in tweet[max(idx-window_size, 0):idx+window_size+1]:
                    if wtoken == -1:
                        continue
                    word_word_count[(ctoken, wtoken)] += 1

        data = list(word_word_count.values())
        rows, cols = list(zip(*word_word_count.keys()))

        cooccurrence_matrix_shape = (len(tweets_vocab),) * 2
        cooccurrence_matrix = sps.coo_matrix(
            (data, (rows, cols)),
            shape=cooccurrence_matrix_shape
        )
        cooccurrence_matrix.setdiag(0)

        # PPMI Matrix
        word_totals = np.array(cooccurrence_matrix.sum(axis=0))[0]
        total = word_totals.sum()
        word_probs = word_totals/total
        ppmi = cooccurrence_matrix / total
        ppmi.data /= (word_probs[ppmi.row] * word_probs[ppmi.col])
        ppmi.row = ppmi.row[ppmi.data > 0]
        ppmi.col = ppmi.col[ppmi.data > 0]
        ppmi.data = ppmi.data[ppmi.data > 0]
        ppmi.data = np.log(ppmi.data)
        ppmi = sps.triu(ppmi)

        # Adjacency matrix
        base_word_index = dataset.shape[0]
        adjacency_shape = (base_word_index + len(tweets_vocab),) * 2

        rows = []
        cols = []
        data = []

        for tidx, tweet in enumerate(tfidf_corpus):
            for widx, tfidf_score in tweet:
                rows.append(tidx)
                cols.append(widx + base_word_index)
                data.append(tfidf_score)

        rows.extend(ppmi.row + base_word_index)
        cols.extend(ppmi.col + base_word_index)
        data.extend(ppmi.data)

        adjacency = sps.coo_matrix((data, (rows, cols)), shape=adjacency_shape)
        adjacency.setdiag(1)
        adjacencies["document_word"] = list(zip(adjacency.row, adjacency.col, adjacency.data))

    print("Saving graphs", file=sys.stderr)
    for graph_type, adjacency in adjacencies.items():
        pd.DataFrame(
            adjacency,
            columns=["row", "col", "weight"]
        ).to_csv(
            "{}.{}.csv.gz".format(args.output_basename, graph_type),
            index=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_input",
                        help="Path to the dataset csv file.")
    parser.add_argument("output_basename",
                        help="Basename (path included) to store the outputs")
    parser.add_argument("--graph-document-word",
                        action="store_true",
                        help="Build graph of document words (Yao et al 2019).")
    parser.add_argument("--graph-document-word-window",
                        default=5,
                        help="Word co-occurrence window (Yao et al 2019).",
                        type=int)
    parser.add_argument("--graph-hashtags",
                        action="store_true",
                        help="Build graph of hashtags.")
    parser.add_argument("--graph-mentions",
                        action="store_true",
                        help="Build graph of mentions.")
    parser.add_argument("--graph-ngrams",
                        default=[],
                        help="Build graph of n-grams.",
                        nargs="+",
                        type=int)
    parser.add_argument("--graph-tfidf",
                        default=[],
                        help="Build graph of top k tfidf tokens.",
                        nargs="+",
                        type=int)
    parser.add_argument("--ignore-hashtags",
                        default=[],
                        help="List of hashtag to ignore when building graph.",
                        nargs="+",
                        type=str)
    parser.add_argument("--ignore-mentions",
                        default=[],
                        help="List of mentions to ignore when building graph.",
                        nargs="+",
                        type=str)
    parser.add_argument("--max-docs",
                        default=1.0,
                        help="Maximum fraction of documents for TF-IDF.",
                        type=float)
    parser.add_argument("--min-docs",
                        default=2,
                        help="Minimum document frequency for TF-IDF.",
                        type=int)
    parser.add_argument("--normalize-hashtags",
                        action="store_true",
                        help="Normalize hashtags in tweets.")
    parser.add_argument("--normalize-mentions",
                        action="store_true",
                        help="Normalize mentions in tweets.")
    parser.add_argument("--remove-hashtags",
                        action="store_true",
                        help="Remove hashtags from tweets.")
    parser.add_argument("--remove-links",
                        action="store_true",
                        help="Remove hyperlinks from tweets.")
    parser.add_argument("--remove-mentions",
                        action="store_true",
                        help="Remove mentions from tweets.")
    parser.add_argument("--remove-numeric",
                        action="store_true",
                        help="Remove numeric tokens from tweets.")
    parser.add_argument("--remove-punctuation",
                        action="store_true",
                        help="Remove punctuation symbols from tweets.")
    parser.add_argument("--remove-stopwords",
                        action="store_true",
                        help="Remove stopwords from tweets.")
    parser.add_argument("--reduce-tweet-word-len",
                        action="store_true",
                        help="Reduce the lenght of words in TweetTokenizer.")
    parser.add_argument("--split-hashtags",
                        action="store_true",
                        help="Camel case splitting of hashtags.")
    parser.add_argument("--supervised-only",
                        action="store_true",
                        help="Build data only from labeled corpora.")
    parser.add_argument("--tweet-lowercase",
                        action="store_true",
                        help="Lowercase the tweets.")

    args = parser.parse_args()

    main(args)
