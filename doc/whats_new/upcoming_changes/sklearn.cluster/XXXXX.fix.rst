- :meth:`cluster.FeatureAgglomeration.inverse_transform` now validates its input
  with :func:`~sklearn.utils.check_array`, so it accepts lists, pandas and polars
  DataFrames, including the output of `transform` when
  `set_output(transform="pandas")` is used, instead of raising an error.
  By :user:`Eugenio Zuccarelli <jayzuccarelli>`.
