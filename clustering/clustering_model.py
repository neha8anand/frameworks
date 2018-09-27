"""
Module containing implementation of a K-Means clustering model.
"""
import numpy as np
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords

class MyModel():
    """A clustering model to identify research areas given information about papers:
        - cleans the json dataset
        - Vectorize the raw text into features.
        - Fit a K-Means clustering model to the resulting features.
    """

    def __init__(self, n):
        self._clusterer = KMeans(n)

    def fit_predict(self, X):
        """Return cluster assignments for training data.
        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        Returns
        -------
        y: The fit model predictions.
        """
        self.X = X
        return self._clusterer.fit_predict(X)

    def predict(self, X):
        """Return cluster assignments for new data."""
        return self._clusterer.predict(X)

    def top_n_features(self, vocabulary, n):
        """Returns top n features for a given vocabulary object (Eg. vectorizer.vocabulary_)."""
        reverse_vocab = reverse_vocabulary(vocabulary)
        centroids = self._clusterer.cluster_centers_ # topics Kmeans has discovered
        indices = np.argsort(centroids, axis=1)
        top_n_indices = indices[:, -n:]
        top_n_features = np.array([reverse_vocab[index] for row in top_n_indices for index in row])
        top_n_features = top_n_features.reshape(len(centroids), -1) # topics with the top n greatest representation in each of the centroids
        return top_n_features

    def most_similar(self, search_text, vectorizer, top_n=5):
        """Returns top n most similar professors for a given search text."""
        x = vectorizer.transform(search_text)
        similarities = cosine_similarity(x, self.X)
        pairs = enumerate(similarities[0])
        most_similar = sorted(pairs, key=lambda item: item[1])[:top_n]
        return np.array(most_similar)

def reverse_vocabulary(vocabulary):
    """Reverses the vocabulary dictionary as returned by the vectorizer."""
    reverse_vocab = {}
    for key, value in vocabulary.items():
        reverse_vocab[value] = key
    return reverse_vocab
