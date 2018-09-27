"""
Module containing model fitting code that implements a sklearn topic model(Choice between LDA, NMF and LSI).
"""
import numpy as np
import pickle
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn

from nltk.corpus import stopwords
stopwords_ = stopwords.words('english')

class MyTopicModel():
    """A topic model to identify topics:
        - Vectorize the raw text into features.
        - Fit a topic model to the resulting features.
    """

    def __init__(self, n_topics=12, algorithm='NMF'):

        if algorithm == 'LDA':
            # Build a Latent Dirichlet Allocation Model
            self._model = LatentDirichletAllocation(n_components=n_topics, max_iter=10, learning_method='online')
        elif algorithm == 'NMF':
            # Build a Non-Negative Matrix Factorization Model
            self._model = NMF(n_components=n_topics)
        elif algorithm == 'LSI':
            # Build a Latent Semantic Indexing Model
            self._model = TruncatedSVD(n_components=n_topics)

    def fit_transform(self, X):
        """Return transformed training data."""
        self.X = X
        self.transformed_X = self._model.fit_transform(X)
        return self.transformed_X

    def transform(self, X):
        """Return transformed new data."""
        return self._model.transform(X)

    def top_n_features(self, vectorizer, top_n=10):
        """Returns top n features/words for all topics given a vectorizer object."""
        topic_words = []
        for _, topic in enumerate(self._model.components_):
            topic_words.append([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-top_n - 1:-1]])
        return np.array(topic_words)

    def most_similar(self, search_text, vectorizer, top_n=5):
        """Returns most similar professors for a given search text (cleaned and tokenized)."""
        x = self._model.transform(vectorizer.transform(clean_text(search_text,stopwords_)))[0]
        similarities = cosine_similarity(x.reshape(1, -1), self.transformed_X)
        pairs = enumerate(similarities[0])
        most_similar = sorted(pairs, key=lambda item: item[1], reverse=True)[:top_n]
        return np.array(most_similar)

    def visualize_lda_model(self, matrix, vectorizer, **kwargs):
        """Visualize LDA model using pyLDAvis"""
        vis = pyLDAvis.sklearn.prepare(self._model, matrix, vectorizer, **kwargs)
        return vis

def get_corpus(filename):
    """Load raw data from a file and return vectorizer and feature_matrix.
    Parameters
    ----------
    filename: The path to a json file containing the university database.

    Returns
    -------
    corpus: A numpy array containing abstracts and titles.
    """
    df_cleaned = database_cleaner(filename)

    # For nlp, only retaining faculty_name, research_areas, paper_titles, abstracts
    df_filtered = df_cleaned[['faculty_name', 'research_areas', 'paper_titles', 'abstracts']]
    missing = df_filtered['paper_titles'] == ''
    df_nlp = df_filtered[~missing]

    # Choosing abstracts and paper_titles to predict topics for a professor
    df_nlp['research_areas'] = df_nlp['research_areas'].apply(lambda x: " ".join(x))
    corpus = (df_nlp['paper_titles'] + df_nlp['abstracts'] + df_nlp['research_areas']).values

    return corpus

def vectorize_corpus(corpus, tf_idf=True, stem_lem=None, **kwargs):
    """
    Parameters
    ----------
    corpus: A numpy array containing abstracts and titles.

    Returns
    -------
    vectorizer: A numpy array containing TfidfVectorizer or CountVectorizer object.
    matrix: The feature matrix (numpy 2-D array) returned by the vectorizer object.
    """

    vectorizer, matrix = feature_matrix(corpus, tf_idf=True, stem_lem=None, **kwargs)

    return vectorizer, matrix