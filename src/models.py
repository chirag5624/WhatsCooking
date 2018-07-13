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
