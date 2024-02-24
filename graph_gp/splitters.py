# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

import itertools

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def split_train_test_indices_stratified_fold(
    n_splits: int, random_state: int, i: int, indices: np.ndarray, y: np.ndarray = None
):
    """
    This function perform a stratified K-fold of the indices according to y using n_splits and returns the i-th split.
    Uses a specific random state.
    Taking i in range 0, ..., n_splits-1 will give all the succesive train-test folds.
    If a stratified K-fold is impossible or y is not specified, uses a K-fold instead.

    Args:
        n_splits: The number of splits of the K-fold.
        random_state: The random state that determines the n_split folds.
        i: The number of the fold to return. i should be less than n_splits.
        indices: The list of indices to split.
        y: The output scalars used to startify the splits.
    Returns:
        indices_train (np.ndarray): The train indices of the i-th fold.
        indices_test (np.ndarray): The test (or valid) indices of the i-th fold.
    """
    if y is None:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    try:
        indices_train, indices_test = next(
            itertools.islice(splitter.split(indices, y), i, None)
        )
    except ValueError:
        print(
            f"Caution, n_splits={n_splits} cannot be greater than the number of members in each class. Using a KFold instead."
        )
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        indices_train, indices_test = next(
            itertools.islice(splitter.split(indices, indices), i, None)
        )
    return indices_train, indices_test
