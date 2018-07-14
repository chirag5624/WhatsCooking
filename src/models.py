import numpy as np
from sklearn.linear_model.logistic import LogisticRegression


class Model(object):
    """Define base class for prediction models"""
    def __init__(self, name):
        self.name = name

    def train(self, train_x, train_y):
        """Train the prediction model"""

    def test(self, test_x):
        """Function to predict and generate file for submission."""


class LogisticClassifier(Model):
    """Multi-label logistic classifier class."""

    def __init__(self, epochs):
        super(LogisticClassifier, self).__init__("logistic regression")
        self.max_epochs = epochs
        self.lr = LogisticRegression(max_iter=epochs)

    def train(self, train_x, train_y):
        print "Training {} model.......".format(self.name)
        self.lr.fit(train_x, train_y)
        print "Training complete!!"

    def test(self, test_x):
        test_y = self.lr.predict(test_x)
        print "Successfully generated predictions for test data."
        return test_y


class LogisticClassifier2(Model):

    def __init__(self, epochs):
        super(LogisticClassifier2, self).__init__("self-written Logistic Regression")
        self.max_epochs = epochs
        self.labels = None
        self.W = None
        self.b = None

    def train(self, train_x, train_y, alpha=0.05):
        num_classes = train_y.nunique()
        self.labels = train_y.unique()
        train_x = train_x.as_matrix()
        m, n = train_x.shape
        print m, n
        print "Number of training examples: {}, number of features: {}".format(m, n)
        self.W = np.random.rand(num_classes, n)
        self.b = np.random.rand(num_classes, 1)
        for i, label in enumerate(self.labels):
            print "training model for class {}".format(label)
            y = np.zeros((m,), dtype='int8')
            y[train_y == label] = 1
            # now train binary logistic regression for this class
            for j in range(self.max_epochs):
                print "epoch #:{}".format(j+1)

                # forward pass
                z = np.matmul(self.W[i], train_x.T) + self.b[i]
                a = 1 / (1 + np.exp(-z))

                # backward pass gradient descent
                dz = y-a
                dw = -1.0*np.matmul(dz, train_x)/m
                db = -1.0*np.mean(dz)
                self.W[i] -= alpha * dw
                self.b[i] -= alpha * db

    def test(self, test_x):
        test_x = test_x.as_matrix()
        predictions = np.matmul(self.W, test_x.T) + self.b
        predictions = np.argmax(predictions, axis=0)
        test_y = self.labels[predictions]
        return test_y
