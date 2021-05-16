import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HandPoseTransform(BaseEstimator, TransformerMixin):
    """
            Custom SKLearn Pipeline Transform function for hand pose data.

            Use the wrist landmark as the origin of the coordinate system.  Shift all hand positions
            to be relative to the wrist.  Scale the max landmark position to have a 3D length = 1
            from the wrist-origin.

            Assumes input data is tabular with each landmark's x, y, & z coords flattened into
            three separate columns.  Also assumes wrist position's z coord is already nearly 0.

    """
    def __init__(self):
        pass

    def fit(self, X, y=0):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        def _shift_and_scale(row):
            # Subtract off wrist position
            row[0::3] = row[0::3] - row[0]
            row[1::3] = row[1::3] - row[1]
            # z origin is already anchored to the wrist, so no need to shift

            # Scale so that the max extension of any hand landmark from the wrist is 1.
            norm = np.max(
                np.sqrt(np.square(row[0::3]) + np.square(row[1::3]) + np.square(row[2::3]))
            )
            row = row / norm
            return row

        return np.apply_along_axis(_shift_and_scale, 1, X_)
