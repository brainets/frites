Performance
-----------

Powerful measures of information (like the mutual information) can detect almost every possible relation between the brain data and an external variable. However, this ability often comes at the cost of being computationally demanding, especially once combined with permutation based statistics. To mitigate those computational issues, we implemented and combined three tricks (see also the `performance section <https://brainets.github.io/frites/auto_examples/index.html#performance>`_ in the examples):

1. **Tensor implementation to avoid nested for loops :** many estimators of information implemented in Frites have a tensor implementation which means that they can operate on multi-dimensional arrays instead of vector-based computations inside nested for loops.
2. **Parallel computing :** we are using the `Joblib <https://joblib.readthedocs.io/en/latest/>`_ library (same as scikit-learn and MNE-Python) for handling parallel computing and using all the cores of the system. Massive computational gains can be observed especially on large computer cluster. We are using the same convention as Scikit-learn i.e. `n_jobs=1` for single core computations, `n_jobs=2` for two cores and `n_jobs=-1` for using all of them.
3. **Python code compilation :** some python code are compiled to machine code using `Numba <http://numba.pydata.org/>`_, for example see the :class:`frites.estimator.BinMIEstimator`
