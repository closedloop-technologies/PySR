import numpy as np
from scipy import signal
from scipy import datasets
# datasets.download_all()
from tempfile import mkdtemp

import pysr
from pysr import PySRRegressor

def load_data():
    # Generate Data for the tutorial
    ascent = datasets.ascent()
    scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                    [-10+0j, 0+ 0j, +10 +0j],
                    [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
    grad = signal.convolve2d(ascent, scharr, boundary='symm', mode='same')

    # Since the above image is 512x512, we can use it as the input data for symbolic regression.
    # We will use the gradient orientation as the target data for symbolic regression.

    # Two create 4 samples we will selct the 4 quadrants of the image
    # The target data will be the gradient orientation
    X = np.array([ascent[:256, :256], ascent[:256, 256:], ascent[256:, :256], ascent[256:, 256:]])
    ascent_shifted = 2 * ascent - 1 
    Y = np.array([ascent_shifted[:256, :256], ascent_shifted[:256, 256:], ascent_shifted[256:, :256], ascent_shifted[256:, 256:]])

    # Expand so that second dimension is 1 so that the image is counted as 1 feature
    X = X.reshape(X.shape[0],1, *X.shape[1:])

    # Expand so that the second dimenion is 1 so that the output only fits one equation
    Y = Y.reshape(Y.shape[0],1, *Y.shape[1:])

    return X, Y


def test_pysr_simple():
    X = np.linspace(-1, 1, 100)[:, np.newaxis]
    Y = np.sin(X) + np.random.normal(0, 0.1, 100)
    
    model = PySRRegressor(
        niterations=30,
        binary_operators=["+", "*"],
        unary_operators=["exp", "sin"],
        populations=10,
        timeout_in_seconds=10,
        model_selection="best",
        tempdir=mkdtemp(prefix='pysr-simple-')
    )
    model.fit(X, Y)
    print(model.get_best()[0].equation)

def test_pysr_multiequation():
    X = np.linspace(-1, 1, 100)[:, np.newaxis]
    Y = np.vstack(
        [
            np.sin(X-2)[:,0] + np.random.normal(0, 0.1, 100),
            1 + 2*np.cos(X)[:,0] + np.random.normal(0, 0.1, 100)
        ]
    ).T

    model = PySRRegressor(
        niterations=30,
        binary_operators=["+", "*"],
        unary_operators=["exp", "sin", "cos"],
        populations=10,
        timeout_in_seconds=10,
        model_selection="best",
        tempdir=mkdtemp(prefix='pysr-multieq-')
    )
    model.fit(X, Y)
    print(model.get_best()[0].equation)
    print(model.get_best()[1].equation)



def test_pysr_multivariate():
    X, Y = load_data()
    model = PySRRegressor(
        niterations=30,
        binary_operators=["+", "*"],
        # unary_operators=["cos", "exp", "sin"],
        populations=10,
        timeout_in_seconds=60,
        model_selection="best",
        tempdir=mkdtemp(prefix='pysr-mv-')
    )
    model.fit(X, Y)


if __name__ == "__main__":
    # test_pysr_simple()
    # test_pysr_multiequation()
    test_pysr_multivariate()
