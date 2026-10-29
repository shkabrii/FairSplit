import numpy as np
from sklearn import metrics
from statsmodels.distributions.empirical_distribution import ECDF


def _pairwise_max(values):
    return max(
        (abs(a - b) for i, a in enumerate(values) for b in values[i + 1:]),
        default=0.0,
    )


def _per_class_max_groupwise(score_per_group, num_classes, s):
    s = s.ravel()
    groups = np.unique(s)
    classes = range(num_classes) if num_classes is not None else [None]
    return max(
        (
            _pairwise_max([score_per_group(k, s == g) for g in groups if np.any(s == g)])
            for k in classes
        ),
        default=0.0,
    ) * 100


def _per_class_max_pairwise_ecdf(probs_per_group_class, num_classes, s, pair_fn):
    s = s.ravel()
    groups = np.unique(s)
    classes = range(num_classes) if num_classes is not None else [None]
    return max(
        (
            pair_fn(probs_per_group_class(k, groups[i]), probs_per_group_class(k, groups[j]))
            for k in classes
            for i in range(len(groups))
            for j in range(i + 1, len(groups))
            if len(probs_per_group_class(k, groups[i])) > 0
            and len(probs_per_group_class(k, groups[j])) > 0
        ),
        default=0.0,
    ) * 100


def _abcc_pair(yi, yj):
    x = np.linspace(0, 1, 10000)
    return np.trapz(np.abs(ECDF(yi)(x) - ECDF(yj)(x)), x)


def _mcdp_pair(yi, yj, support):
    return np.max(np.abs(ECDF(yi)(support) - ECDF(yj)(support)))


def demographic_parity(y_pred, s, num_classes=None):
    if num_classes is None:
        return _per_class_max_groupwise(lambda _, m: (y_pred[m] > 0.5).mean(), None, s)
    y_class = np.argmax(y_pred, axis=1)
    return _per_class_max_groupwise(lambda k, m: (y_class[m] == k).mean(), num_classes, s)


def equal_opportunity(y_pred, y_gt, s, num_classes=None):
    y_gt = y_gt.ravel()
    if num_classes is None:
        m = y_gt == 1
        return demographic_parity(y_pred[m], s.ravel()[m], None)
    y_class = np.argmax(y_pred, axis=1)
    s = s.ravel()

    def rate(k, group_mask):
        cond = group_mask & (y_gt == k)
        return (y_class[cond] == k).mean() if np.any(cond) else 0.0

    return _per_class_max_groupwise(rate, num_classes, s)


def equalized_odds(y_pred, y_gt, s, num_classes=None):
    y_gt, s = y_gt.ravel(), s.ravel()
    if num_classes is None:
        y_class = (y_pred > 0.5).astype(int)
        classes = [1]
    else:
        y_class = np.argmax(y_pred, axis=1)
        classes = range(num_classes)
    groups = np.unique(s)

    def per_class_max(k):
        tpr = {g: (y_class[(s == g) & (y_gt == k)] == k).mean()
               for g in groups if np.any((s == g) & (y_gt == k))}
        fpr = {g: (y_class[(s == g) & (y_gt != k)] == k).mean()
               for g in groups if np.any((s == g) & (y_gt != k))}
        return max(
            (abs(tpr[a] - tpr[b]) + abs(fpr[a] - fpr[b])
             for i, a in enumerate(groups) for b in groups[i + 1:]
             if a in tpr and b in tpr and a in fpr and b in fpr),
            default=0.0,
        )

    return max((per_class_max(k) for k in classes), default=0.0) * 100


def abcc(y_pred, s, num_classes=None):
    if num_classes is None:
        s_arr = s.ravel()
        return _per_class_max_pairwise_ecdf(
            lambda _, g: y_pred[s_arr == g], None, s, _abcc_pair
        )
    s_arr = s.ravel()
    return _per_class_max_pairwise_ecdf(
        lambda k, g: y_pred[s_arr == g, k], num_classes, s, _abcc_pair
    )


def mcdp0(y_pred, s, num_classes=None):
    s = s.ravel()
    groups = np.unique(s)
    classes = range(num_classes) if num_classes is not None else [None]

    def support(k):
        return y_pred if k is None else y_pred[:, k]

    def per_group(k, g):
        col = y_pred if k is None else y_pred[:, k]
        return col[s == g]

    return max(
        (
            _mcdp_pair(per_group(k, groups[i]), per_group(k, groups[j]), support(k))
            for k in classes
            for i in range(len(groups))
            for j in range(i + 1, len(groups))
            if np.any(s == groups[i]) and np.any(s == groups[j])
        ),
        default=0.0,
    ) * 100


def metric_evaluation(y_gt, y_pre, s, prefix="", metrics_to_compute=None):
    y_gt, s = y_gt.ravel(), s.ravel()
    is_multiclass = y_pre.ndim == 2 and y_pre.shape[1] > 1
    num_classes = y_pre.shape[1] if is_multiclass else None

    if is_multiclass:
        y_pred_class = np.argmax(y_pre, axis=1)
    else:
        y_pre = y_pre.ravel()
        y_pred_class = (y_pre > 0.5).astype(int)

    if metrics_to_compute is None:
        metrics_to_compute = ["acc", "ap", "f1", "dp", "abcc", "mcdp0", "eopp", "eodd"]

    funcs = {
        "acc": lambda: metrics.accuracy_score(y_gt, y_pred_class) * 100,
        "f1": lambda: metrics.f1_score(
            y_gt, y_pred_class,
            average='macro' if is_multiclass else 'binary',
            zero_division=0,
        ) * 100,
        "dp":    lambda: demographic_parity(y_pre, s, num_classes),
        "abcc":  lambda: abcc(y_pre, s, num_classes),
        "mcdp0": lambda: mcdp0(y_pre, s, num_classes),
        "eopp":  lambda: equal_opportunity(y_pre, y_gt, s, num_classes),
        "eodd":  lambda: equalized_odds(y_pre, y_gt, s, num_classes),
    }
    if not is_multiclass:
        funcs["ap"] = lambda: metrics.average_precision_score(y_gt, y_pre) * 100

    result = {}
    for name in metrics_to_compute:
        if name not in funcs:
            continue
        key = f"{prefix}/{name}" if prefix else name
        result[key] = funcs[name]()
    return result
