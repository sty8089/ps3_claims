import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        """Initialize the Winsorizer.
        
        Parameters
        ----------
        lower_quantile : float, default=0.05
            Lower quantile for clipping (e.g., 0.05 means 5th percentile)
        upper_quantile : float, default=0.95
            Upper quantile for clipping (e.g., 0.95 means 95th percentile)
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        """Fit the Winsorizer by computing quantile values from X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : None
            Ignored, present for sklearn compatibility
            
        Returns
        -------
        self : object
            Fitted transformer
        """
        # Convert to numpy array if needed
        X = np.asarray(X)
        
        # Compute the actual quantile values from the data
        self.lower_quantile_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile, axis=0)
        
        return self

    def transform(self, X):
        """Transform X by clipping values at the fitted quantiles.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_clipped : ndarray of shape (n_samples, n_features)
            Transformed data with values clipped at quantiles
        """
        # Check that fit has been called
        check_is_fitted(self, ['lower_quantile_', 'upper_quantile_'])
        
        # Convert to numpy array if needed
        X = np.asarray(X)
        
        # Clip the values between lower and upper quantiles
        X_clipped = np.clip(X, self.lower_quantile_, self.upper_quantile_)
        
        return X_clipped
