"""
Module containing model fitting code for a classification model.
"""
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,  AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import log_loss, confusion_matrix, precision_recall_curve, precision_score, recall_score, roc_auc_score

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

class MyModel():
    """A churn prediction model:
        - Fit a classifier model to the resulting features.
    """

    def __init__(self, classifier): 
        self._classifier = classifier

    def fit(self, X, y):
        """Fit a churn prediction classifier model.
        Parameters
        ----------
        X: A pandas dataframe, to be used as predictors.
        y: A pandas dataframe, to be used as responses.
        Returns
        -------
        self: The fit model object.
        """
        self._classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        """Make probability predictions on new data."""
        return self._classifier.predict_proba(X)

    def predict(self, X):
        """Return the class on new data."""
        return self._classifier.predict(X)

    def score(self, X, y):
        """Return the classification accuracy score on new data."""
        return self._classifier.score(X, y)
    
    def calc_precision_score(self, X, y):
        """Return the classification precision score on new data."""
        y_pred = self.predict(X)
        return precision_score(y, y_pred)
    
    def calc_recall_score(self, X, y):
        """Return the classification recall score on new data."""
        y_pred = self.predict(X)
        return recall_score(y, y_pred)
    
    def calc_roc_auc_score(self, X, y):
        """Return the classification roc auc score on new data."""
        y_prob = self.predict_proba(X)[:, 1] # probability of positive class
        return roc_auc_score(y, y_prob)

    def roc_curve(self, probabilities, labels):
        '''
        Return the True Positive Rates, False Positive Rates and Thresholds for the
        ROC curve given predicted probabilities and true labels.
        '''
        thresholds = np.sort(probabilities)

        tprs = []
        fprs = []

        num_positive_cases = sum(labels)
        num_negative_cases = len(labels) - num_positive_cases

        for threshold in thresholds:
            predicted_positive = probabilities >= threshold
            true_positives = np.sum(predicted_positive * labels)
            false_positives = np.sum(predicted_positive) - true_positives
            tpr = true_positives / num_positive_cases
            fpr = false_positives / num_negative_cases
            fprs.append(fpr)
            tprs.append(tpr)
        return tprs, fprs, thresholds.tolist()

    def draw_precision_recall_curve(self, X, y, save_path=None):
        """Return the ROC or precision recall curve."""
        y_prob = self.predict_proba(X)[:, 1] # probability of positive class
        tpr, fpr, thresholds = self.roc_curve(y_prob, y)
        plt.title('Receiver Operating Characteristic Plot')
        plt.plot(fpr, tpr, 'b',label='AUC = %0.5f'% roc_auc_score(y, y_prob))
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.1,1.1])
        plt.ylim([-0.1,1.1])
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity, Recall)")
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def calc_confusion_matrix(self, X, y):
        """Return the classification confusion matrix on new data.
        Returns
        -------
        confusion_matrix  : ndarray - 2D, with values corresponding to:
                                          -----------
                                          | TN | FP |
                                          -----------
                                          | FN | TP |
                                          -----------
        """
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def draw_confusion_matrix(self, X,  y, save_path=None):
        """Return the plot of the classification confusion matrix on new data."""
        cm = self.calc_confusion_matrix(X, y)
        fig, ax = plt.subplots()
        im = ax.matshow(cm)
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
            
        plt.title('Confusion matrix')
        fig.colorbar(im)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_feature_importances(self, feature_names, top_n=15):
        """Return a plot of the feature importances for top_n features."""
        feat_scores = pd.DataFrame({'Fraction of Samples Affected' : self._classifier.feature_importances_},
                           index=feature_names)
        feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected')[-top_n:]
        feat_scores.plot(kind='barh')

    def optimize_model_params(self, X, y, grid, scoring='roc_auc'):                            
        """Optimize model parameters using GridSearchCV."""
        gridsearch = GridSearchCV(self._classifier,
                                  grid,
                                  n_jobs=-1,
                                  scoring=scoring)

        gridsearch.fit(X, y)
        return gridsearch.best_estimator_
                    
def cross_val(classifiers, X, y, scoring='roc_auc', cv=5):
    """Returns train and test scores for classifiers using K-fold cross validation."""
    scores = np.array([cross_val_score(classifier, X, y, scoring=scoring, cv=cv, n_jobs=-1) for classifier in list(classifiers.values())])
    return pd.DataFrame(scores.T, columns=list(classifiers.keys()))

def calc_cost_benefit_matrix(user_worth=100, discount=20):
        """Return cost-benefit matrix for a given user worth and yearly discount%.
        Parameters
        ----------
        user_worth    : ndarray - 1D
        discount      : ndarray - 1D

        Returns
        -------
        cost_benefit  : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TN | FP |
                                          -----------
                                          | FN | TP |
                                          -----------
        """
        # Assuming the baseline was that nobody gets a discount.
        tn, fn = 0, 0
        fp = 0 - (user_worth * discount / 100)
        tp = user_worth - (user_worth * discount / 100)
        return np.array([[tn, fp], [fn, tp]])

def profit_curve(cost_benefit_matrix, predicted_probs, labels):
    """Function to calculate list of profits based on supplied cost-benefit
    matrix and prediced probabilities of data points and their true labels.
    
    Parameters
    ----------
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TN | FP |
                                          -----------
                                          | FN | TP |
                                          -----------
    predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
    labels          : ndarray - 1D, true label of datapoints, 0 or 1
    
    Returns
    -------
    profits    : ndarray - 1D
    thresholds : ndarray - 1D
    """
    n_obs = len(labels)
    thresholds = np.unique(sorted(predicted_probs, reverse=True))
    profits = []

    for threshold in thresholds:
        y_predict = predicted_probs >= threshold
        cm = confusion_matrix(labels, y_predict)
        threshold_profit = np.sum(cm * cost_benefit_matrix) / n_obs
        profits.append(threshold_profit)
    return np.array(profits), thresholds

def calc_model_profits(models, cost_benefit_matrix, X, y):
    """Function to calculate profits for multiple models.
    
    Parameters
    ----------
    models          : ndarray - 1D, classifier model objects.
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TN | FP |
                                          -----------
                                          | FN | TP |
                                          -----------
    
    X               : ndarray - 2D
    y               : ndarray - 1D
    Returns
    -------
    model_profits    : list((model, profits, thresholds))
    """
    list_profits, list_thresholds = [], []
    # # Correcting for imbalanced classes in training set using SMOTE
    # sm = SMOTE(random_state=2)
    # X_sm, y_sm = sm.fit_sample(X, y.ravel())

    for model in models:
        model.fit(X, y)
        predicted_probs = model.predict_proba(X)[:, 1]
        profits, thresholds = profit_curve(cost_benefit_matrix, predicted_probs, y)
        list_profits.append(profits)
        list_thresholds.append(thresholds)
    return list(zip(models, list_profits, list_thresholds))


def plot_model_profits(model_profits, save_path=None):
    """Plotting function to compare profit curves of different models.
    Parameters
    ----------
    model_profits : list((model, profits, thresholds))
    save_path     : str, file path to save the plot to. If provided plot will be
                         saved and not shown.
    """
    for model, profits, _ in model_profits:
        percentages = np.linspace(0, 100, profits.shape[0])
        plt.plot(percentages, profits, label=model.__class__.__name__)

    plt.title("Profit Curves")
    plt.xlabel("Percentage of train instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def find_best_threshold(model_profits):
    """Find model-threshold combo that yields highest profit.
    Parameters
    ----------
    model_profits : list((model, profits, thresholds))
    Returns
    -------
    max_model     : str
    max_threshold : float
    max_profit    : float
    """
    max_model = None
    max_threshold = None
    max_profit = None
    for model, profits, thresholds in model_profits:
        max_index = np.argmax(profits)
        if not max_model or profits[max_index] > max_profit:
            max_model = model
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
    return max_model, max_threshold, max_profit
    