from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

def z_score_outliers(df, feature):
    if df[feature].std == 0:
        return 0
    
    low = df[feature].mean() - 3 * df[feature].std()
    high = df[feature].mean() + 3 * df[feature].std()

    return df[(df[feature] > high) | (df[feature] < low)]

def iqr_features(df, feature):
    if df[feature].std() == 0:
        return 0
    
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)

    IQR = Q3 - Q1

    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR

    return df[(df[feature] > high) | (df[feature] < low)]

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, iqr_feats):
        self.iqr_feats = iqr_feats
        self.bounds = {}

    def fit(self, X, y=None):
        for feature in self.iqr_feats:
            Q1 = X[feature].quantile(0.25)
            Q3 = X[feature].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds[feature] = {
                'low': Q1 - 1.5 * IQR,
                'high': Q3 - 1.5 * IQR
            }
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for feature in self.iqr_feats:
            bounds = self.bounds[feature]
            X_copy[feature] = np.clip(X_copy[feature], bounds['low'], bounds['high'])

        return X_copy
