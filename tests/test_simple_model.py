from unittest import TestCase
import matplotlib.pyplot as plt
import numpy as np

from learna.model import Model


def plot_decision_boundary(pred_func, X, Y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
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


def load_planar_dataset(m):
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype="uint8")  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


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

    def test_simple_classifier_2(self):
        np.random.seed(1)

        # m_test = 200
        m_train = 400
        # TODO(Research):  Calculations fail when using high number of n_h or num_iterations
        n_h = 4
        num_iterations = 10000
        X, Y = load_planar_dataset(m_train)

        model = Model()
        fitted_parameters = model.fit(X, Y, n_h, num_iterations=num_iterations, print_cost=True)

        # X_test = np.random.rand(n_x, m_test)
        # Y_test = np.where(np.sum(X, axis=0) > n_x * 0.5, 1, 0).reshape(n_y, m_test)
        # Y_predicted = model.predict(parameters=fitted_parameters, X=X_test)

        plot_decision_boundary(lambda x: model.predict(fitted_parameters, x.T), X, Y)
