from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from ml_from_scratch.naive_bayes import  NaiveBayes
from ml_from_scratch.utils import accuracy_score


def main():
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    # tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    # X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    X_test_counts = count_vect.transform(twenty_test.data)
    # X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    nb_clf = NaiveBayes()
    nb_clf.fit(X_train_counts, twenty_train.target)
    predicted = nb_clf.predict(X_test_counts)
    accuracy = accuracy_score(twenty_test.target, predicted)
    print('Accuracy: %s' % accuracy)

    sklearn_clf = MultinomialNB().fit(X_train_counts, twenty_train.target)
    sklearn_predicted = sklearn_clf.predict(X_test_counts)
    sklearn_accuracy = accuracy_score(twenty_test.target, sklearn_predicted)
    print('Accuracy from sklearn: %s' % sklearn_accuracy)


if __name__ == '__main__':
    main()
