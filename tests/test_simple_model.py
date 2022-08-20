from unittest import TestCase
import matplotlib.pyplot as plt
import numpy as np

from learna.model import Model


def plot_decision_boundary(pred_func, X, Y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)

    plt.show()


class TestModel(TestCase):
    def test_simple_classifier(self):
        np.random.seed(2)

        # m_test = 200
        m_train = 400
        n_x = 2
        # TODO(Research):  Calculations fail when using high number of n_h or num_iterations
        n_h = 4
        n_y = 1
        num_iterations = 10000
        X = np.random.rand(n_x, m_train)
        Y = np.where(np.sum(X, axis=0) > n_x * 0.5, 1, 0).reshape(n_y, m_train)

        model = Model()
        fitted_parameters = model.fit(X, Y, n_h, num_iterations=num_iterations, print_cost=True)

        # X_test = np.random.rand(n_x, m_test)
        # Y_test = np.where(np.sum(X, axis=0) > n_x * 0.5, 1, 0).reshape(n_y, m_test)
        # Y_predicted = model.predict(parameters=fitted_parameters, X=X_test)

        plot_decision_boundary(lambda x: model.predict(fitted_parameters, x.T), X, Y)
