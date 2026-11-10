"""Original cost-sensitive XGBoost notebook (ds004504, AD / FTD / HC).

Per 4 s epoch: slowing ratio + 4 relative band powers for 19 channels (95
features). Subject-grouped 5-fold CV (GroupKFold), per-fold StandardScaler and
SelectKBest(k=45), balanced sample weights, subject-level soft voting. Also
runs the single 80/20 subject hold-out (GroupShuffleSplit, random_state 42).
"""
import numpy as np
from _common import parser, save

from cybex_eeg.dementia import ds004504 as D
from cybex_eeg.dementia.models import NOTEBOOK_XGB, subject_vote_groupkfold
from cybex_eeg.dementia.qeeg import epoch_slowing_features

if __name__ == "__main__":
    import mne
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    from xgboost import XGBClassifier

    p = parser(__doc__)
    p.add_argument("--bids-root", required=True)
    a = p.parse_args()
    labels = D.load_participants(a.bids_root).set_index("subject_id")["diagnosis"]
    X, y, g = [], [], []
    for sub, path in D.derivative_recordings(a.bids_root):
        raw = D.load_recording(path, 0.5, 30.0)
        ep = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose=False)
        psd = ep.compute_psd(method="welch", fmin=0.5, fmax=30.0, verbose=False)
        P, f = psd.get_data(return_freqs=True)
        F = epoch_slowing_features(P, f)
        X.append(F); y += [labels[D.canonical_id(sub)]] * len(F); g += [sub] * len(F)
    X, g = np.vstack(X), np.array(g)
    le = LabelEncoder(); y = le.fit_transform(y)
    cvres = subject_vote_groupkfold(X, y, g)
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=a.seed).split(X, y, g))
    sc = StandardScaler().fit(X[tr]); sel = SelectKBest(f_classif, k=45).fit(sc.transform(X[tr]), y[tr])
    cls = np.unique(y[tr]); w = dict(zip(cls, compute_class_weight("balanced", classes=cls, y=y[tr])))
    m = XGBClassifier(**NOTEBOOK_XGB).fit(sel.transform(sc.transform(X[tr])), y[tr],
                                         sample_weight=np.array([w[c] for c in y[tr]]))
    proba = m.predict_proba(sel.transform(sc.transform(X[te])))
    subs = np.unique(g[te])
    hold = float(np.mean([y[te][g[te] == s][0] == proba[g[te] == s].mean(0).argmax() for s in subs]))
    print(f"5-fold subject-level accuracy: {cvres['subject_accuracy']:.4f} | hold-out ({len(subs)} subjects): {hold:.4f}")
    save({"classes": le.classes_.tolist(), "groupkfold": cvres, "holdout_accuracy": hold,
          "holdout_subjects": subs.tolist(), "test_accuracy": cvres["subject_accuracy"]}, "subject_vote_xgb", a.outdir)
