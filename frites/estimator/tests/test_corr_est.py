"""Test correlation and distance correlation estimators."""
import numpy as np

from frites.estimator import CorrEstimator, DcorrEstimator


array_equal = np.testing.assert_array_equal


class TestCorrEstimator(object):

    def test_corr_definition(self):
        """Test definition of correlation estimator."""
        for method in ['pearson', 'spearman']:
            for implementation in ['vector', 'tensor']:
                CorrEstimator(method=method, implementation=implementation)

    def test_corr_estimate(self):
        """Test getting the core function."""
        x, y = np.random.rand(10, 1, 100), np.random.rand(10, 1, 100)
        cat = np.array([0] * 50 + [1] * 50)
        est = CorrEstimator()

        for func in [0, 1]:
            if func == 0:    # estimator.get_function()
                fcn = est.get_function()
            elif func == 1:  # estimator.estimate
                fcn = est.estimate

            # no categories
            array_equal(fcn(x[0, 0, :], y[0, 0, :]).shape, (1, 1))
            array_equal(fcn(x[0, :, :], y[0, 0, :]).shape, (1, 1))
            array_equal(fcn(x, y).shape, (1, 10))

            # with categories
            array_equal(fcn(x[0, 0, :], y[0, 0, :],
                            categories=cat).shape, (2, 1))
            array_equal(fcn(x[0, :, :], y[0, 0, :],
                            categories=cat).shape, (2, 1))
            array_equal(fcn(x, y, categories=cat).shape, (2, 10))

    def test_corr_implementation(self):
        """Compare the results of the implementations"""
        # generate random data
        x, y = np.random.rand(10, 1, 100), np.random.rand(10, 1, 100)
        cat = np.array([0] * 50 + [1] * 50)

        # define estimators
        pear_vec = CorrEstimator(method='pearson', implementation='vector')
        pear_ten = CorrEstimator(method='pearson', implementation='tensor')
        spear_vec = CorrEstimator(method='spearman', implementation='vector')
        spear_ten = CorrEstimator(method='spearman', implementation='tensor')

        for cate in [None, cat]:
            # pearson correlation
            corr_vec = pear_vec.estimate(x, y, categories=cate)
            corr_ten = pear_ten.estimate(x, y, categories=cate)
            np.testing.assert_array_almost_equal(corr_vec, corr_ten)

            # spearman correlation
            corr_vec = spear_vec.estimate(x, y, categories=cate)
            corr_ten = spear_ten.estimate(x, y, categories=cate)
            np.testing.assert_array_almost_equal(corr_vec, corr_ten)

    def test_corr_functional(self):
        """Functional test of the correlation."""
        fcn = CorrEstimator().get_function()

        # no categories
        x, y = np.random.rand(2, 1, 100), np.random.rand(100)
        x[1, ...] += y.reshape(1, -1)
        corr = fcn(x, y).ravel()
        assert corr[0] < corr[1]

        # with categories
        x, y = np.random.rand(100), np.random.rand(100)
        cat = np.array([0] * 50 + [1] * 50)
        x[0:50] += y[0:50]
        x[50::] -= y[50::]
        corr_nocat = fcn(x, y).ravel()
        corr_cat = fcn(x, y, categories=cat).ravel()
        assert (corr_nocat < corr_cat[0]) and (corr_nocat < abs(corr_cat[1]))
        assert (corr_cat[0] > 0) and (corr_cat[1] < 0)

    def test_dcorr_definition(self):
        """Test definition of distance correlation estimator."""
        DcorrEstimator(implementation='auto')
        DcorrEstimator(implementation='frites')
        DcorrEstimator(implementation='dcor')

    def test_dcorr_estimate(self):
        """Test getting the core function."""
        x, y = np.random.rand(10, 1, 100), np.random.rand(10, 1, 100)
        cat = np.array([0] * 50 + [1] * 50)

        for imp in ['auto', 'frites', 'dcor']:
            est = DcorrEstimator(implementation=imp)
            for func in [0, 1]:
                # function definition
                if func == 0:    # estimator.get_function()
                    fcn = est.get_function()
                elif func == 1:  # estimator.estimate
                    fcn = est.estimate

                # no categories
                array_equal(fcn(x[0, 0, :], y[0, 0, :]).shape, (1, 1))
                array_equal(fcn(x[0, :, :], y[0, 0, :]).shape, (1, 1))
                array_equal(fcn(x, y).shape, (1, 10))

                # with categories
                array_equal(fcn(x[0, 0, :], y[0, 0, :],
                                categories=cat).shape, (2, 1))
                array_equal(fcn(x[0, :, :], y[0, 0, :],
                                categories=cat).shape, (2, 1))
                array_equal(fcn(x, y, categories=cat).shape, (2, 10))

    def test_dcorr_functional(self):
        """Functional test of the correlation."""
        for imp in ['auto', 'frites', 'dcor']:
            fcn = DcorrEstimator(implementation=imp).get_function()

            # no categories
            x, y = np.random.rand(2, 1, 100), np.random.rand(100)
            x[1, ...] += y.reshape(1, -1)
            dcorr = fcn(x, y).ravel()
            assert dcorr[0] < dcorr[1]

            # with categories
            x, y = np.random.rand(100), np.random.rand(100)
            cat = np.array([0] * 50 + [1] * 50)
            x[0:50] += y[0:50]
            x[50::] -= y[50::]
            dc_nocat = fcn(x, y).ravel()
            dc_cat = fcn(x, y, categories=cat).ravel()
            assert (dc_nocat < dc_cat[0]) and (dc_nocat < dc_cat[1])
            assert (0 < dc_cat[0]) and (0 < dc_cat[1])


if __name__ == '__main__':
    TestCorrEstimator().test_corr_implementation()
