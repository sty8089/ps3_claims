import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):
    """Test that Winsorizer correctly clips values at specified quantiles."""
    # Create test data
    np.random.seed(42)  # For reproducibility
    X = np.random.normal(0, 1, 1000).reshape(-1, 1)  # Shape: (1000, 1)
    
    # Initialize and fit the winsorizer
    winsorizer = Winsorizer(lower_quantile=lower_quantile, upper_quantile=upper_quantile)
    winsorizer.fit(X)
    
    # Transform the data
    X_transformed = winsorizer.transform(X)
    
    # Test 1: Check that fitted quantiles are stored correctly
    assert hasattr(winsorizer, 'lower_quantile_'), "lower_quantile_ not set during fit"
    assert hasattr(winsorizer, 'upper_quantile_'), "upper_quantile_ not set during fit"
    
    # Test 2: Check that the quantiles are computed correctly
    expected_lower = np.quantile(X, lower_quantile, axis=0)
    expected_upper = np.quantile(X, upper_quantile, axis=0)
    np.testing.assert_allclose(winsorizer.lower_quantile_, expected_lower)
    np.testing.assert_allclose(winsorizer.upper_quantile_, expected_upper)
    
    # Test 3: Check that all values are within the quantile bounds
    assert np.all(X_transformed >= winsorizer.lower_quantile_), "Values below lower quantile found"
    assert np.all(X_transformed <= winsorizer.upper_quantile_), "Values above upper quantile found"
    
    # Test 4: Check that shape is preserved
    assert X_transformed.shape == X.shape, "Shape changed after transformation"
    
    # Test 5: For the (0, 1) case, data should be unchanged
    if lower_quantile == 0 and upper_quantile == 1:
        np.testing.assert_array_equal(X, X_transformed, 
                                       err_msg="Data changed when using (0, 1) quantiles")
    
    # Test 6: For the (0.5, 0.5) case, all values should equal the median
    if lower_quantile == 0.5 and upper_quantile == 0.5:
        # All values should be the same (the median)
        assert np.allclose(X_transformed, X_transformed[0]), \
            "All values should be equal (the median) for (0.5, 0.5)"
