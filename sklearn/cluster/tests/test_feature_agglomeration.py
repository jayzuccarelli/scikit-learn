"""
Tests for sklearn.cluster._feature_agglomeration
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn.cluster import FeatureAgglomeration
from sklearn.datasets import make_blobs
from sklearn.utils._testing import _convert_container, assert_array_almost_equal


def test_feature_agglomeration():
    n_clusters = 1
    X = np.array([0, 0, 1]).reshape(1, 3)  # (n_samples, n_features)

    agglo_mean = FeatureAgglomeration(n_clusters=n_clusters, pooling_func=np.mean)
    agglo_median = FeatureAgglomeration(n_clusters=n_clusters, pooling_func=np.median)
    agglo_mean.fit(X)
    agglo_median.fit(X)

    assert np.size(np.unique(agglo_mean.labels_)) == n_clusters
    assert np.size(np.unique(agglo_median.labels_)) == n_clusters
    assert np.size(agglo_mean.labels_) == X.shape[1]
    assert np.size(agglo_median.labels_) == X.shape[1]

    # Test transform
    Xt_mean = agglo_mean.transform(X)
    Xt_median = agglo_median.transform(X)
    assert Xt_mean.shape[1] == n_clusters
    assert Xt_median.shape[1] == n_clusters
    assert Xt_mean == np.array([1 / 3.0])
    assert Xt_median == np.array([0.0])

    # Test inverse transform
    X_full_mean = agglo_mean.inverse_transform(Xt_mean)
    X_full_median = agglo_median.inverse_transform(Xt_median)
    assert np.unique(X_full_mean[0]).size == n_clusters
    assert np.unique(X_full_median[0]).size == n_clusters

    assert_array_almost_equal(agglo_mean.transform(X_full_mean), Xt_mean)
    assert_array_almost_equal(agglo_median.transform(X_full_median), Xt_median)


def test_feature_agglomeration_feature_names_out():
    """Check `get_feature_names_out` for `FeatureAgglomeration`."""
    X, _ = make_blobs(n_features=6, random_state=0)
    agglo = FeatureAgglomeration(n_clusters=3)
    agglo.fit(X)
    n_clusters = agglo.n_clusters_

    names_out = agglo.get_feature_names_out()
    assert_array_equal(
        [f"featureagglomeration{i}" for i in range(n_clusters)], names_out
    )


@pytest.mark.parametrize("constructor_name", ["list", "array", "pandas", "polars"])
def test_feature_agglomeration_inverse_transform_array_like(constructor_name):
    """Check that `inverse_transform` accepts any array-like input."""
    X, _ = make_blobs(n_features=6, random_state=0)
    agglo = FeatureAgglomeration(n_clusters=3).fit(X)
    Xt = agglo.transform(X)
    expected = agglo.inverse_transform(Xt)

    Xt_container = _convert_container(Xt, constructor_name)
    assert_array_equal(agglo.inverse_transform(Xt_container), expected)


@pytest.mark.parametrize("constructor_name", ["list", "array"])
def test_feature_agglomeration_inverse_transform_1d(constructor_name):
    """Check that `inverse_transform` accepts 1-D input of shape (n_clusters,)."""
    X, _ = make_blobs(n_features=6, random_state=0)
    agglo = FeatureAgglomeration(n_clusters=3).fit(X)
    Xt = agglo.transform(X)
    expected = agglo.inverse_transform(Xt)

    Xt_1d = _convert_container(Xt[0], constructor_name)
    assert_array_equal(agglo.inverse_transform(Xt_1d), expected[0])


def test_feature_agglomeration_inverse_transform_set_output_pandas():
    """Check that `inverse_transform` accepts the output of `transform` when
    `set_output(transform="pandas")` is used."""
    pytest.importorskip("pandas")
    X, _ = make_blobs(n_features=6, random_state=0)
    agglo = FeatureAgglomeration(n_clusters=3).set_output(transform="pandas")
    Xt = agglo.fit_transform(X)
    X_inv = agglo.inverse_transform(Xt)
    assert isinstance(X_inv, np.ndarray)
    assert X_inv.shape == X.shape
    assert_array_almost_equal(agglo.transform(X_inv).to_numpy(), Xt.to_numpy())
