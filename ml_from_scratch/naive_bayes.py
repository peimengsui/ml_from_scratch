import numpy as np
import math


class NaiveBayes:
    """
    Multinomial Naive Bayes Classifier
    """
    def fit(self, X, y , alpha=1):
        self.X, self.y = X, y
        self.X = self.X.toarray()
        self.alpha = alpha
        self.classes = np.unique(y)
        self.parameters = dict()
        for c in self.classes:
            X_where_c = self.X[np.where(y == c)]
            col_sum = X_where_c.sum(axis=0)
            total_sum = X_where_c.sum()
            for feature_idx in range(X_where_c.shape[1]):
                self.parameters[(c, feature_idx)] = (col_sum[feature_idx]+self.alpha) / \
                                                    (total_sum + self.alpha*X_where_c.shape[1])

    def _calculate_prior(self, c):
        """ Calculate the prior of class c
        (samples where class == c / total number of samples)"""
        frequency = np.mean(self.y == c)
        return frequency

    def _classify(self, sample):
        """ Classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X),
            or Posterior = Likelihood * Prior / Scaling Factor
        P(Y|X) - The posterior is the probability that sample x is of class y given the
                 feature values of x being distributed according to distribution of y and the prior.
        P(X|Y) - Likelihood of data X given class distribution Y.
        P(Y)   - Prior (given by _calculate_prior)
        P(X)   - Scales the posterior to make it a proper probability distribution.
                 This term is ignored in this implementation since it doesn't affect
                 which class distribution the sample is most likely to belong to.
        Classifies the sample as the class that results in the largest P(Y|X) (posterior)
        """
        posteriors = []
        # Go through list of classes
        for c in self.classes:
            # Initialize posterior as prior
            posterior = np.log(self._calculate_prior(c))
            # Naive assumption (independence):
            # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
            # Posterior is product of prior and likelihoods (ignoring scaling factor)
            for feature_idx, feature_value in enumerate(sample.toarray().ravel()):
                # Likelihood of feature value given distribution of feature values given y
                log_likelihood = feature_value * np.log(self.parameters[(c, feature_idx)])
                posterior += log_likelihood
            posteriors.append(posterior)
        # Return the class with the largest posterior probability
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """ Predict the class labels of the samples in X """
        y_pred = [self._classify(sample) for sample in X]
        return y_pred
