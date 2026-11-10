"""Classifier protocols for the ds004504 papers."""
from __future__ import annotations

import numpy as np

P7_GRID = {
    "n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0], "colsample_bytree": [0.8, 1.0], "min_child_weight": [1, 3, 5],
}  # 216 combinations

NOTEBOOK_XGB = dict(objective="multi:softprob", num_class=3, n_estimators=300, learning_rate=0.03,
                    max_depth=4, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.5,
                    random_state=42, n_jobs=-1)


def xgb_gridsearch(X, y, grid=P7_GRID, random_state: int = 42, nested: bool = True, n_jobs: int = -1) -> dict:
    """RobustScaler + XGBoost tuned by GridSearchCV under stratified 5-fold CV.

    Returns GridSearchCV's ``best_score_`` (the figure reported in the
    label-correction paper) and, when ``nested`` is True, a nested-CV estimate
    in which tuning is repeated inside each outer fold. The nested estimate is
    the unbiased one; ``best_score_`` is optimistic because the same folds are
    used to choose and to score the hyper-parameters.
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from xgboost import XGBClassifier

    pipe = Pipeline([("scale", RobustScaler()),
                     ("xgb", XGBClassifier(random_state=random_state, eval_metric="mlogloss", n_jobs=1))])
    pgrid = {f"xgb__{k}": v for k, v in grid.items()}
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state)
    gs = GridSearchCV(pipe, pgrid, cv=cv, scoring="accuracy", n_jobs=n_jobs).fit(X, y)
    out = {"best_score": float(gs.best_score_),
           "best_params": {k.replace("xgb__", ""): v for k, v in gs.best_params_.items()}}
    if nested:
        inner = StratifiedKFold(5, shuffle=True, random_state=random_state + 1)
        outer = StratifiedKFold(5, shuffle=True, random_state=random_state)
        s = cross_val_score(GridSearchCV(pipe, pgrid, cv=inner, scoring="accuracy", n_jobs=n_jobs), X, y, cv=outer)
        out["nested_cv_mean"], out["nested_cv_sd"] = float(s.mean()), float(s.std())
    return out


def subject_vote_groupkfold(X, y, groups, k_best: int = 45, n_splits: int = 5) -> dict:
    """Original cost-sensitive XGBoost notebook: epoch-level GroupKFold by subject,
    StandardScaler and SelectKBest(k=45) fitted per fold, balanced sample
    weights, then subject-level soft voting (mean probability across epochs)."""
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    from xgboost import XGBClassifier

    t, p, subs = [], [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        sel = SelectKBest(f_classif, k=min(k_best, X.shape[1])).fit(sc.transform(X[tr]), y[tr])
        Xtr, Xte = sel.transform(sc.transform(X[tr])), sel.transform(sc.transform(X[te]))
        cls = np.unique(y[tr])
        w = dict(zip(cls, compute_class_weight("balanced", classes=cls, y=y[tr])))
        params = dict(NOTEBOOK_XGB, num_class=len(np.unique(y)))
        m = XGBClassifier(**params).fit(Xtr, y[tr], sample_weight=np.array([w[c] for c in y[tr]]))
        proba = m.predict_proba(Xte)
        for s in np.unique(groups[te]):
            ix = groups[te] == s
            t.append(y[te][ix][0]); p.append(int(proba[ix].mean(0).argmax())); subs.append(s)
    t, p = np.array(t), np.array(p)
    return {"subject_accuracy": float((t == p).mean()), "y_true": t.tolist(), "y_pred": p.tolist(),
            "subjects": [str(s) for s in subs]}
