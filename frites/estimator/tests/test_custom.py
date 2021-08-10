"""Test custom estimator."""
import numpy as np

from frites.estimator import CustomEstimator


class TestCustomEstimator(object):

    def test_definition(self):
        # test overall definition
        def fcn(x, y): return (x + y).mean(axis=(1, 2))  # noqa
        CustomEstimator('test', 'cc', fcn)

        # test three outputs
        try:
            def fcn(x, y, z): return (x + y).mean(axis=(1, 2))  # noqa
            CustomEstimator('test', 'cc', fcn)
        except TypeError:
            pass

        # test numpy array type
        try:
            def fcn(x, y): return (x + y).mean(axis=(1, 2)).tolist()  # noqa
            CustomEstimator('test', 'cc', fcn)
        except TypeError:
            pass

        # test output shape
        try:
            def fcn(x, y): return (x + y).mean(axis=(0, 2))  # noqa
            CustomEstimator('test', 'cc', fcn)
        except ValueError:
            pass

    def test_estimation(self):
        # define estimator
        def fcn(x, y): return (x + y).mean(axis=(1, 2))  # noqa
        est = CustomEstimator('test', 'cc', fcn)

        # define variables
        x = np.random.rand(10, 5, 200)
        y = np.random.rand(10, 5, 200)
        cat = np.array([0] * 100 + [1] * 100)

        for c in [None, cat]:
            est.estimate(x[0, 0, :], y[0, 0, :], categories=c)
            est.estimate(x[0, :, :], y[0, :, :], categories=c)
            est.estimate(x, y, categories=c)
            est.estimate(x, y[0, 0, :], categories=c)
            est.estimate(x, y[0, :, :], categories=c)

        # test 4d
        x_4d = np.random.rand(3, 10, 5, 200)
        y_1 = y[0, 0, :]
        assert est.estimate(x_4d, y_1).shape == (1, 3, 10)
        assert est.estimate(x_4d, y_1, categories=cat).shape == (2, 3, 10)


if __name__ == '__main__':
    TestCustomEstimator().test_estimation()
