"""Module to explore data.

Contains functions to help study, visualize and understand datasets.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import plotly.offline as py
import plotly.graph_objs as go

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


def get_num_classes(labels):
    """Gets the total number of classes.

    # Arguments
        labels: list, label values.
            There should be at lease one sample for values in the
            range (0, num_classes -1)

    # Returns
        int, total number of classes.

    # Raises
        ValueError: if any label value in the range(0, num_classes - 1)
            is missing or if number of classes is <= 1.
    """
    num_classes = max(labels) + 1
    missing_classes = [i for i in range(num_classes) if i not in labels]
    if len(missing_classes):
        raise ValueError(
            "Missing samples with label value(s) "
            "{missing_classes}. Please make sure you have "
            "at least one sample for every label value "
            "in the range(0, {max_class})".format(
                missing_classes=missing_classes, max_class=num_classes - 1
            )
        )

    if num_classes <= 1:
        raise ValueError(
            "Invalid number of labels: {num_classes}."
            "Please make sure there are at least two classes "
            "of samples".format(num_classes=num_classes)
        )
    return num_classes


def get_num_words_per_sample(sample_texts):
    """Gets the median number of words per sample given corpus.

    # Arguments
        sample_texts: list, sample texts.

    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)


def plot_frequency_distribution_of_ngrams(
    sample_texts, ngram_range=(1, 2), num_ngrams=50
):
    """Plots the frequency distribution of n-grams.

    # Arguments
        samples_texts: list, sample texts.
        ngram_range: tuple (min, mplt), The range of n-gram values to consider.
            Min and mplt are the lower and upper bound values for the range.
        num_ngrams: int, number of n-grams to plot.
            Top `num_ngrams` frequent n-grams will be plotted.
    """
    # Create args required for vectorizing.
    kwargs = {
        "ngram_range": (1, 1),
        "dtype": "int32",
        "strip_accents": "unicode",
        "decode_error": "replace",
        "analyzer": "word",  # Split text into word tokens.
    }
    vectorizer = CountVectorizer(**kwargs)

    # This creates a vocabulary (dict, where keys are n-grams and values are
    # idxices). This also converts every text to an array the length of
    # vocabulary, where every element idxicates the count of the n-gram
    # corresponding at that idxex in vocabulary.
    vectorized_texts = vectorizer.fit_transform(sample_texts)

    # This is the list of all n-grams in the index order from the vocabulary.
    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(num_ngrams, len(all_ngrams))
    # ngrams = all_ngrams[:num_ngrams]

    # Add up the counts per n-gram ie. column-wise
    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
    all_counts, all_ngrams = zip(
        *[(c, n) for c, n in sorted(zip(all_counts, all_ngrams), reverse=True)]
    )
    ngrams = list(all_ngrams)[:num_ngrams]
    counts = list(all_counts)[:num_ngrams]

    trace = go.Bar(x=ngrams, y=counts)
    layout = go.Layout(
        title="Frequency distribution of n-grams",
        xaxis=dict(title="N-grams"),
        yaxis=dict(title="Frequencies"),
    )
    fig = dict(data=[trace], layout=layout)
    py.iplot(fig)


def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    if not isinstance(sample_texts, dict):
        sample_texts = dict(sample=sample_texts)

    traces = list()
    for title, sample_text in sample_texts.items():
        trace = go.Histogram(x=[len(s) for s in sample_text], name=title, opacity=0.75)
        traces.append(trace)

    layout = go.Layout(
        title="Sample length distrobution",
        barmode="overlay",
        xaxis=dict(title="Length of a sample"),
        yaxis=dict(title="Frequency"),
    )
    fig = dict(data=traces, layout=layout)
    py.iplot(fig)


def plot_class_distribution(labels):
    """Plots the class distribution.

    # Arguments
        labels: list, label values.
            There should be at lease one sample for values in the
            range (0, num_classes -1)
    """
    num_classes = get_num_classes(labels)
    count_map = Counter(labels)
    counts = [count_map[i] for i in range(num_classes)]
    idx = np.arange(num_classes)

    trace = go.Bar(x=idx, y=counts)
    layout = go.Layout(
        title="Sample length distrobution",
        barmode="overlay",
        xaxis=dict(title="Length of a sample"),
        yaxis=dict(title="Class distribution"),
    )
    fig = dict(data=[trace], layout=layout)
    py.iplot(fig)
