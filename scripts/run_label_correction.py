"""Label correction + 39-feature XGBoost on OpenNeuro ds004504 (AD / FTD / HC).

1. Build one feature row per subject: 11 graph-spectral features (from the
   original pipeline's table via --gsp-features, or recomputed with
   cybex_eeg.dementia.gsp) + 28 qEEG band features.
2. Join features to diagnoses by participant identifier, with the Algorithm 1
   checks, and report how many labels a positional join would have changed.
3. Six-classifier benchmark under stratified 5-fold CV (RobustScaler in-fold).
4. GridSearchCV-tuned XGBoost (216-point grid): best_score_ and a nested-CV
   estimate.
5. Selection-leakage demonstration: top-15 ANOVA features chosen on all
   subjects before CV versus inside each fold.
"""
import numpy as np
import pandas as pd
from _common import parser, save

from cybex_eeg.dementia import ds004504 as D
from cybex_eeg.dementia.gsp import GSP_FEATURES, gsp_features
from cybex_eeg.dementia.models import xgb_gridsearch
from cybex_eeg.dementia.qeeg import qeeg28


def build_table(bids_root, gsp_csv=None):
    rows = []
    for sub, path in D.derivative_recordings(bids_root):
        raw = D.load_recording(path, l_freq=0.5, h_freq=45.0)
        data, sf = raw.get_data(), raw.info["sfreq"]
        q, qnames = qeeg28(data, sf, raw.ch_names)
        row = {"participant_id": sub, **dict(zip(qnames, q))}
        if gsp_csv is None:
            row.update(dict(zip(GSP_FEATURES, gsp_features(data))))
        rows.append(row)
        print(f"{sub}: done")
    feats = pd.DataFrame(rows)
    if gsp_csv is not None:
        g = pd.read_csv(gsp_csv)
        idcol = next(c for c in g.columns if c.lower() in ("participant_id", "subject", "subject_id", "file", "filename"))
        g["participant_id"] = g[idcol].map(D.canonical_id)
        num = g.select_dtypes("number").columns.difference(["epoch_id", "MMSE", "Age"])
        g = g.groupby("participant_id")[list(num)].mean().reset_index()  # epoch-level table -> per subject
        feats = feats.merge(g, on="participant_id", how="inner", validate="one_to_one")
    return feats


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, RobustScaler
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    p = parser(__doc__)
    p.add_argument("--bids-root", required=True)
    p.add_argument("--gsp-features", default=None, help="CSV of the original 11 graph features")
    p.add_argument("--features-out", default="features_39.csv")
    p.add_argument("--no-nested", action="store_true")
    p.add_argument("--expected-n", type=int, default=88, help="subjects expected after the merge (88 for ds004504)")
    p.add_argument("--quick-grid", action="store_true", help="2-point grid (for smoke tests only)")
    a = p.parse_args()

    parts = D.load_participants(a.bids_root)
    feats = build_table(a.bids_root, a.gsp_features)
    feats.to_csv(a.features_out, index=False)
    merged = D.keyed_merge(feats, parts, expected_n=a.expected_n)
    # What a positional join would have assigned, with rows in the order they were actually built.
    pos = D.positional_merge(feats, parts)
    changed = int((pos["participant_id_x"].map(D.canonical_id) != pos["subject_id"]).sum()) \
        if "participant_id_x" in pos else int((feats["participant_id"].map(D.canonical_id).values
                                               != parts["subject_id"].values[:len(feats)]).sum())
    feature_cols = [c for c in feats.columns if c != "participant_id"]
    X = merged[feature_cols].to_numpy(float)
    y = LabelEncoder().fit_transform(merged["diagnosis"])
    print("Feature matrix:", X.shape, "classes:", np.bincount(y))
    cv = StratifiedKFold(5, shuffle=True, random_state=a.seed)
    bench = {}
    for name, clf in {"knn": KNeighborsClassifier(), "random_forest": RandomForestClassifier(random_state=a.seed),
                      "xgboost": XGBClassifier(random_state=a.seed, eval_metric="mlogloss"),
                      "logistic_regression": LogisticRegression(max_iter=5000),
                      "svm": SVC(), "naive_bayes": GaussianNB()}.items():
        s = cross_val_score(make_pipeline(RobustScaler(), clf), X, y, cv=cv)
        bench[name] = {"mean": float(s.mean()), "sd": float(s.std())}
        print(f"{name:20s} {s.mean():.4f} ± {s.std():.4f}")
    grid = {"n_estimators": [50], "max_depth": [2, 3]} if a.quick_grid else None
    tuned = xgb_gridsearch(X, y, random_state=a.seed, nested=not a.no_nested, **({"grid": grid} if grid else {}))
    print("GridSearchCV best_score_:", tuned["best_score"], tuned["best_params"])
    leaky_idx = SelectKBest(f_classif, k=15).fit(RobustScaler().fit_transform(X), y).get_support()
    leaky = cross_val_score(make_pipeline(RobustScaler(), XGBClassifier(random_state=a.seed)), X[:, leaky_idx], y, cv=cv)
    clean = cross_val_score(make_pipeline(RobustScaler(), SelectKBest(f_classif, k=15),
                                          XGBClassifier(random_state=a.seed)), X, y, cv=cv)
    save({"n_subjects": int(len(y)), "n_features": int(X.shape[1]), "feature_names": feature_cols,
          "rows_misassigned_by_positional_join": changed,
          "six_classifier_benchmark": bench, "xgb_gridsearch": tuned,
          "top15_selection_outside_cv": float(leaky.mean()), "top15_selection_inside_cv": float(clean.mean())},
         "label_correction", a.outdir)
