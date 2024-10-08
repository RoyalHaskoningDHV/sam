from typing import Callable, Any

import numpy as np
from sklearn.utils import check_random_state


def iter_shuffled(X, columns_to_shuffle=None, pre_shuffle=False, random_state=None):
    """
    Return an iterator of X matrices which have one or more columns shuffled.
    After each iteration yielded matrix is mutated inplace, so
    if you want to use multiple of them at the same time, make copies.

    ``columns_to_shuffle`` is a sequence of column numbers to shuffle.
    By default, all columns are shuffled once, i.e. columns_to_shuffle
    is ``range(X.shape[1])``.

    If ``pre_shuffle`` is True, a copy of ``X`` is shuffled once, and then
    result takes shuffled columns from this copy. If it is False,
    columns are shuffled on fly. ``pre_shuffle = True`` can be faster
    if there is a lot of columns, or if columns are used multiple times.
    """
    rng = check_random_state(random_state)

    if columns_to_shuffle is None:
        columns_to_shuffle = range(X.shape[1])

    if pre_shuffle:
        X_shuffled = X.copy()
        rng.shuffle(X_shuffled)

    X_res = X.copy()
    for columns in columns_to_shuffle:
        if pre_shuffle:
            X_res[:, columns] = X_shuffled[:, columns]
        else:
            rng.shuffle(X_res[:, columns])
        yield X_res
        X_res[:, columns] = X[:, columns]


def _get_scores_shufled(score_func, X, y, columns_to_shuffle=None, random_state=None):
    """
    Return the scores of the shuffled features
    """
    Xs = iter_shuffled(X, columns_to_shuffle, random_state=random_state)
    return np.array([score_func(X_shuffled, y) for X_shuffled in Xs])


def get_score_importances(
    score_func: Callable[[Any, Any], float],
    X: Any,
    y: Any,
    n_iter: int = 5,
    columns_to_shuffle=None,
    random_state=None,
):
    """
    Returns a tuple (base_score, score_decreases) with the base score and
    score decreases when a feature is not available.

    ``base_score`` is ``score_func(X, y)``; ``score_decreases``
    is a list of length ``n_iter`` with feature importance arrays
    (each array is of shape ``n_features``); feature importances are computed
    as score decrease when a feature is not available.

    ``n_iter`` iterations of the basic algorithm is done, each iteration
    starting from a different random seed.
    """
    rng = check_random_state(random_state)
    base_score = score_func(X, y)
    scores_decreases = []
    for i in range(n_iter):
        scores_shuffled = _get_scores_shufled(
            score_func, X, y, columns_to_shuffle=columns_to_shuffle, random_state=rng
        )
        scores_decreases.append(-scores_shuffled + base_score)
    return base_score, scores_decreases
