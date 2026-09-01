!pip -q install ucimlrepo

# ======================================================
# Gumbel λU Feature Selection
# ======================================================
# Baselines: Mutual Information (MI), L1/ElasticNet (L1EN), mRMR, ReliefF
# Our method: Gumbel λU (tail dependence)
# Extras: λU bootstrap CI (B=1000), stability, perturbations (light), k-sweep
# Models: RF, XGB, LR, GB
# Plots & CSVs saved to ./outputs/<DATASET>/
# ======================================================

import os, sys, time, json, random, glob, re, warnings
from pathlib import Path
from collections import defaultdict

# Headless plotting (safe on Windows/PyCharm)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr, norm

# Data
from ucimlrepo import fetch_ucirepo

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix, balanced_accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import permutation_importance


from xgboost import XGBClassifier
from joblib import Parallel, delayed

# stats for McNemar
from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------
# Global Config (EXTENDED MODE)
# -------------------------------
CFG = {
    # CORE
    "SEED": 123,
    "TEST_SIZE": 0.20,
    "VAL_SIZE": 0.20,               # from TRAIN for threshold tuning
    "TOPK": 7,                      # dataset-specific overrides below
    "EVAL_MODELS": ["RF", "XGB", "LR", "GB"],

    # λU bootstrap (CI)
    "BOOTSTRAP_LAM": True,
    "N_BOOT_LAM": 1000,             # supervisor requirement

    # Stability
    "STABILITY": True,
    "STABILITY_B": 1000,

    # k-sweep (dataset-specific overrides below)
    "DO_K_SWEEP": True,
    "K_SWEEP": [5],

    # Perturbations (light)
    "PERTURB": True,
    "ROB_NOISE_SD": [0.0, 0.10, 0.20, 0.35, 0.50],    # baseline included as 0.0
    "ROB_LABEL_FLIP": [0.00, 0.05, 0.10, 0.20, 0.30],
    "ROB_MCAR": [0.00, 0.10, 0.20, 0.35, 0.50],
    "ROB_SUBSAMPLE": [1.00, 0.75, 0.50],        # keep full (change to 0.70 to test subsample)

    # Calibrate & class weights
    "CALIBRATE": True,
    "USE_SMOTE": False,

    # Threshold selection
    "THRESH_CRITERION": "f1",       # "f1" or "youden"

    # Parallelism / IO
    "N_JOBS": 2,                    # 1–2 recommended to avoid memmap weirdness
    "OUT_ROOT": "./outputs",
    "PRIMARY_METRIC": "ROC_AUC",

    # PIMA hygiene (zeros-as-missing)
    "PIMA_TREAT_ZEROS_AS_MISSING": True,
    "PIMA_ZERO_MISS_COLS": ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"],

}

# -------------------------------
# Reproducibility
# -------------------------------
def set_seed(seed=123):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# -------------------------------
# Small timer helper
# -------------------------------
class Timer:
    def __init__(self): self.t, self._last = {}, {}
    def start(self, key): self._last[key] = time.perf_counter()
    def stop(self, key):  self.t[key] = self.t.get(key, 0.0) + (time.perf_counter() - self._last.get(key, time.perf_counter()))
    def dump(self, out_csv): pd.Series(self.t).to_csv(out_csv)

# -------------------------------
# Utilities
# -------------------------------
def to_pseudo_observations(arr):
    return rankdata(arr, method='average') / (len(arr) + 1.0)

def gumbel_theta_from_tau(tau, eps=1e-6):
    if tau is None or np.isnan(tau):
        return None
    # clip to (0, 1 - eps); allow tiny negatives to map to None (no upper-tail dep)
    if tau <= 0:
        return None
    if tau >= 1 - eps:
        return 1.0 / eps  # ~ very large theta → lambda_U ~ 1
    return 1.0 / (1.0 - tau)

def gumbel_lambda_U_from_theta(theta):
    if theta is None or theta < 1.0: return 0.0
    return 2.0 - 2.0**(1.0 / theta)

def class_imbalance_report(y, name="dataset"):
    pos = int(np.sum(y == 1)); neg = int(np.sum(y == 0)); n = int(len(y))
    ratio = f"{pos}:{neg}" if neg > 0 else "NA"
    return {"name": name, "n": n, "pos": pos, "neg": neg,
            "pos_rate": pos / n if n > 0 else np.nan, "ratio_pos_neg": ratio}

from sklearn.impute import SimpleImputer

def pima_zero_impute_train_only(X_tr, X_te, outdir, cols, enable=True):
    """
    Treat zeros as missing (NaN) in the listed columns for PIMA.
    Fit a median imputer on TRAIN ONLY; transform both TRAIN and TEST.
    Writes an audit CSV with zero counts before imputation.
    """
    if not enable:
        return X_tr, X_te

    X_tr = X_tr.copy()
    X_te = X_te.copy()
    cols = [c for c in cols if c in X_tr.columns]
    if not cols:
        return X_tr, X_te

    # audit zeros before
    def zero_counts(df):
        return {c: int((df[c] == 0).sum()) for c in cols}
    before = {"train": zero_counts(X_tr), "test": zero_counts(X_te)}

    # zeros -> NaN
    for c in cols:
        X_tr.loc[X_tr[c] == 0, c] = np.nan
        X_te.loc[X_te[c] == 0, c] = np.nan

    # median imputation fit on TRAIN only
    imp = SimpleImputer(strategy="median")
    X_tr[cols] = imp.fit_transform(X_tr[cols])
    X_te[cols] = imp.transform(X_te[cols])

    # audit file
    try:
        pd.DataFrame.from_records(
            [{"split": s, **before[s]} for s in ["train","test"]]
        ).to_csv(Path(outdir) / "pima_zero_impute_report.csv", index=False)
    except Exception:
        pass

    return X_tr, X_te


def pick_threshold(y_true, y_prob, criterion="f1"):
    if criterion == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        idx = int(np.argmax(tpr - fpr))
        return float(thr[idx])
    best_t = 0.5; best_score = -np.inf
    for t in np.quantile(y_prob, np.linspace(0.01, 0.99, 199)):
        y_hat = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_hat == 1))
        fp = np.sum((y_true == 0) & (y_hat == 1))
        fn = np.sum((y_true == 1) & (y_hat == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_score:
            best_score = f1; best_t = t
    return float(best_t)

def plot_confmat(set_name, model_name, y_true, y_pred, outdir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
    except Exception:
        ax.imshow(cm)
        ax.set_xticks([0,1]); ax.set_xticklabels(["No","Yes"])
        ax.set_yticks([0,1]); ax.set_yticklabels(["No","Yes"])
        for (i,j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha='center', va='center')
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix • {set_name} • {model_name}")
    plt.tight_layout()
    plt.savefig(outdir / f"confmat_{set_name}_{model_name}.png", dpi=300)
    plt.close()

# -------------------------------
# Models
# -------------------------------
def base_model(name, seed, scale_pos_weight):
    if name == 'RF':
        return RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=seed, class_weight="balanced"
        )
    elif name == 'XGB':
        return XGBClassifier(
            eval_metric='logloss', random_state=seed,
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=scale_pos_weight,
            tree_method="hist"  # CPU-fast
        )
    elif name == 'LR':
        return LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced")
    elif name == 'GB':
        return GradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError("Unknown model")

def build_model(name, cfg, seed, scale_pos_weight):
    model = base_model(name, seed, scale_pos_weight)
    if cfg["CALIBRATE"] and name in {'RF', 'GB', 'XGB'}:
        model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    return model

# -------------------------------
# Feature Scoring (Gumbel λU)
# -------------------------------
def _one_feat_lambdaU(col_values, y_train):
    up = to_pseudo_observations(col_values)
    yp = to_pseudo_observations(y_train)
    tau, _ = kendalltau(up, yp)
    if tau is None or np.isnan(tau):
        tau = 0.0
    theta = gumbel_theta_from_tau(tau)
    lamU = gumbel_lambda_U_from_theta(theta)
    return float(tau if tau is not None else 0.0), (float(theta) if theta is not None else None), float(lamU)

def gumbel_lambdaU_feature_scores(X_train: pd.DataFrame, y_train: np.ndarray, n_jobs=1):
    rows = Parallel(n_jobs=n_jobs, prefer="threads", max_nbytes=None)(
        delayed(_one_feat_lambdaU)(X_train[f].values, y_train) for f in X_train.columns
    )
    df = pd.DataFrame({
        "Feature": X_train.columns,
        "Tau": [r[0] for r in rows],
        "Theta": [r[1] for r in rows],
        "Lambda_U": [r[2] for r in rows],
    }).sort_values("Lambda_U", ascending=False)
    return df

def topk_from_df(df, k=5, col="Lambda_U"):
    return df.nlargest(k, col)["Feature"].tolist()

def bootstrap_lambdaU_ci(Xtr, ytr, features, B=1000, seed=123, n_jobs=1):
    rng = np.random.default_rng(seed); n = len(ytr)
    def one_boot(idx):
        Xb, yb = Xtr.iloc[idx], ytr[idx]
        dfb = gumbel_lambdaU_feature_scores(Xb[features], yb, n_jobs=1)
        return [dfb.loc[dfb["Feature"] == f, "Lambda_U"].values[0] for f in features]
    idxs = [rng.integers(0, n, size=n) for _ in range(B)]
    rows = Parallel(n_jobs=n_jobs, prefer="threads", max_nbytes=None)(
        delayed(one_boot)(idx) for idx in idxs
    )
    lam_mat = np.array(rows)
    means = lam_mat.mean(axis=0)
    lo = np.quantile(lam_mat, 0.025, axis=0)
    hi = np.quantile(lam_mat, 0.975, axis=0)
    return means, lo, hi

# -------------------------------
# L1/ElasticNet Selection (embedded)
# -------------------------------
def select_l1_enet(Xtr, ytr, k=5, seed=123, penalty='elasticnet', l1_ratio=0.7):
    lr = LogisticRegression(
        penalty=penalty, l1_ratio=(l1_ratio if penalty=='elasticnet' else None),
        solver='saga', max_iter=4000, random_state=seed, class_weight='balanced'
    )
    scaler = StandardScaler().fit(Xtr)
    lr.fit(scaler.transform(Xtr), ytr)
    coefs = np.abs(lr.coef_).ravel()
    sel = pd.Series(coefs, index=Xtr.columns).sort_values(ascending=False)
    return sel.head(k).index.tolist()

def rank_l1_enet_table(Xtr, ytr, seed=123, penalty='elasticnet', l1_ratio=0.7):
    lr = LogisticRegression(
        penalty=penalty, l1_ratio=(l1_ratio if penalty=='elasticnet' else None),
        solver='saga', max_iter=4000, random_state=seed, class_weight='balanced'
    )
    scaler = StandardScaler().fit(Xtr)
    lr.fit(scaler.transform(Xtr), ytr)
    coefs = np.abs(lr.coef_).ravel()
    return pd.DataFrame({"Feature": Xtr.columns, "Score": coefs}).sort_values("Score", ascending=False)

# -------------------------------
# mRMR (custom, lightweight)
# -------------------------------
def mrmr_select(Xtr, ytr, k=5, seed=123):
    np.random.seed(seed)
    rel = mutual_info_classif(Xtr, ytr, random_state=seed)
    rel_s = pd.Series(rel, index=Xtr.columns)
    selected = []
    remaining = list(Xtr.columns)
    corr = Xtr.corr().abs()
    while len(selected) < min(k, Xtr.shape[1]):
        best_f, best_score = None, -1e9
        for f in remaining:
            red = corr.loc[f, selected].mean() if selected else 0.0
            score = rel_s[f] - red
            if score > best_score:
                best_score, best_f = score, f
        selected.append(best_f)
        remaining.remove(best_f)
    return selected

def mrmr_rank_table(Xtr, ytr, seed=123):
    feats = mrmr_select(Xtr, ytr, k=Xtr.shape[1], seed=seed)
    return pd.DataFrame({"Feature": feats, "Rank": np.arange(1, len(feats)+1)})

# -------------------------------
# ReliefF (binary, lightweight, memory-safe)
# -------------------------------
def relieff_scores(Xtr: pd.DataFrame, ytr: np.ndarray, n_neighbors=10, n_samples=None, seed=123):
    """
    Returns a pandas Series of ReliefF scores indexed by feature name.
    Assumes numeric features (CDC/PIMA are numeric). Standardizes first.
    """
    rng = np.random.default_rng(seed)
    X = Xtr.astype(np.float32).copy()
    mu = X.mean(axis=0).values
    sd = X.std(axis=0, ddof=0).replace(0, 1.0).values
    X = ((X - mu) / sd).to_numpy(dtype=np.float32)
    y = ytr.astype(int)

    n, d = X.shape
    if n_samples is None or n_samples > n:
        n_samples = n

    idx_all = np.arange(n)
    if n_samples < n:
        idx_pos = idx_all[y == 1]
        idx_neg = idx_all[y == 0]
        n_pos = max(1, int(n_samples * (len(idx_pos) / n)))
        n_neg = max(1, n_samples - n_pos)
        samp = np.concatenate([
            rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False),
            rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False)
        ])
        rng.shuffle(samp)
    else:
        samp = idx_all

    X_pos = X[y == 1]; X_neg = X[y == 0]
    nn_pos = NearestNeighbors(n_neighbors=min(n_neighbors, max(1, len(X_pos)-1)), algorithm="brute", metric="euclidean")
    nn_neg = NearestNeighbors(n_neighbors=min(n_neighbors, max(1, len(X_neg)-1)), algorithm="brute", metric="euclidean")
    if len(X_pos) > 0: nn_pos.fit(X_pos)
    if len(X_neg) > 0: nn_neg.fit(X_neg)

    scores = np.zeros(d, dtype=np.float64)

    for i in samp:
        xi = X[i:i+1]
        if y[i] == 1:
            hits   = X_pos[nn_pos.kneighbors(xi, return_distance=False)[0]] if len(X_pos) > 1 else xi
            misses = X_neg[nn_neg.kneighbors(xi, return_distance=False)[0]] if len(X_neg) > 0 else xi
        else:
            hits   = X_neg[nn_neg.kneighbors(xi, return_distance=False)[0]] if len(X_neg) > 1 else xi
            misses = X_pos[nn_pos.kneighbors(xi, return_distance=False)[0]] if len(X_pos) > 0 else xi

        diff_hit   = np.abs(hits - xi).mean(axis=0)
        diff_miss  = np.abs(misses - xi).mean(axis=0)
        scores += (diff_miss - diff_hit)

    scores /= max(1, len(samp))
    return pd.Series(scores, index=Xtr.columns).sort_values(ascending=False)

def relieff_select(Xtr, ytr, k=5, seed=123, n_neighbors=10, n_samples=None):
    sc = relieff_scores(Xtr, ytr, n_neighbors=n_neighbors, n_samples=n_samples, seed=seed)
    return sc.head(k).index.tolist()

def relieff_rank_table(Xtr, ytr, seed=123, n_neighbors=10, n_samples=None):
    sc = relieff_scores(Xtr, ytr, n_neighbors=n_neighbors, n_samples=n_samples, seed=seed)
    return pd.DataFrame({"Feature": sc.index, "Score": sc.values})

# -------------------------------
# Plot helpers
# -------------------------------
def plot_superimposed_roc_by_model(model_name, curves, outdir):
    plt.figure(figsize=(7,5.2))
    for (label, fpr, tpr, auc) in curves:
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'--',lw=1,color='gray')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {model_name} (sets)"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(outdir / f"roc_superimposed_BYMODEL_{model_name}.png", dpi=300); plt.close()

def plot_superimposed_pr_by_model(model_name, curves, outdir):
    plt.figure(figsize=(7,5.2))
    for (label, rec, prec, ap) in curves:
        plt.plot(rec, prec, label=f"{label} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall — {model_name} (sets)"); plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(outdir / f"pr_superimposed_BYMODEL_{model_name}.png", dpi=300); plt.close()

def plot_superimposed_roc_by_set(set_name, curves, outdir):
    plt.figure(figsize=(7,5.2))
    for (label, fpr, tpr, auc) in curves:
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],'--',lw=1,color='gray')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {set_name} (models)"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(outdir / f"roc_superimposed_BYSET_{set_name}.png", dpi=300); plt.close()

def plot_superimposed_pr_by_set(set_name, curves, outdir):
    plt.figure(figsize=(7,5.2))
    for (label, rec, prec, ap) in curves:
        plt.plot(rec, prec, label=f"{label} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision–Recall — {set_name} (models)"); plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(outdir / f"pr_superimposed_BYSET_{set_name}.png", dpi=300); plt.close()

def plot_metric_curve_for_model(df_results, model_name, metric, outdir, order):
    sub = df_results[df_results['Model']==model_name].set_index('Set').reindex(order).dropna().reset_index()
    if sub.empty: return
    plt.figure(figsize=(6.8,4.2))
    plt.plot(sub['Set'], sub[metric], marker='o', linewidth=2)
    plt.ylim(0, 1); plt.xlabel("Feature set"); plt.ylabel(metric.replace('_',' '))
    plt.title(f"{metric.replace('_',' ')} — {model_name}")
    plt.grid(True, alpha=0.25); plt.tight_layout()
    plt.savefig(outdir / f"{metric.lower()}_curve_{model_name}.png", dpi=300); plt.close()

def plot_metric_curve_all_models(df_results, metric, outdir, order, models):
    plt.figure(figsize=(7.2,4.6))
    for m in models:
        sub = df_results[df_results['Model']==m].set_index('Set').reindex(order).dropna().reset_index()
        if sub.empty: continue
        plt.plot(sub['Set'], sub[metric], marker='o', linewidth=2, label=m)
    plt.ylim(0, 1); plt.xlabel("Feature set"); plt.ylabel(metric.replace('_',' '))
    plt.title(f"{metric.replace('_',' ')} — All Models")
    plt.legend(); plt.grid(True, alpha=0.25); plt.tight_layout()
    plt.savefig(outdir / f"{metric.lower()}_curve_ALLMODELS.png", dpi=300); plt.close()

def compute_permutation_importance_for_best(X_tr, X_te, y_tr, y_te,
                                            feats, model_name, cfg, outdir, set_name, n_repeats=500):
    """
    Fits the chosen model on TRAIN (full train used for final fit in your pipeline)
    and computes test-set permutation importance for the provided feature subset.
    Saves a CSV and a bar plot. Does not alter any upstream logic.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build the model with identical settings (including calibration & class weights)
    pos = np.sum(y_tr == 1); neg = np.sum(y_tr == 0)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    model = build_model(model_name, cfg, cfg["SEED"], scale_pos_weight)

    # Match the exact training convention you already use:
    # LR gets standardized arrays; tree models use raw columns.
    if model_name == 'LR':
        scaler = StandardScaler().fit(X_tr[feats])
        Xtr_use = scaler.transform(X_tr[feats])
        Xte_use = scaler.transform(X_te[feats])
    else:
        Xtr_use = X_tr[feats]
        Xte_use = X_te[feats]

    # GB can use sample weights (as in your evaluation)
    sample_w = compute_sample_weight(class_weight='balanced', y=y_tr) if model_name=='GB' else None
    if isinstance(model, CalibratedClassifierCV):
        model.fit(Xtr_use, y_tr, sample_weight=None if model_name!='GB' else sample_w)
    else:
        model.fit(Xtr_use, y_tr, sample_weight=sample_w)

    # Permutation importance scored by ROC AUC (uses predict_proba/decision_function under the hood)
    pi = permutation_importance(
        model, Xte_use, y_te,
        n_repeats=n_repeats,
        random_state=cfg["SEED"],
        n_jobs=cfg["N_JOBS"],
        scoring='roc_auc'
    )

    # Pack results with feature names
    df_pi = pd.DataFrame({
        "Feature": feats,
        "Importance_Mean": pi.importances_mean,
        "Importance_Std":  pi.importances_std
    }).sort_values("Importance_Mean", ascending=False)

    # Save CSV
    df_pi.to_csv(outdir / f"permutation_importance_{set_name}_{model_name}.csv", index=False)

    # Save a simple bar plot (top 20 or all if fewer)
    top = min(20, len(df_pi))
    plt.figure(figsize=(8, 5))
    plt.bar(range(top), df_pi["Importance_Mean"].values[:top],
            yerr=df_pi["Importance_Std"].values[:top], capsize=3)
    plt.xticks(range(top), df_pi["Feature"].values[:top], rotation=30, ha='right')
    plt.ylabel("Permutation importance (ROC AUC drop)")
    plt.title(f"Permutation importance — {set_name} • {model_name} (n_repeats={n_repeats})")
    plt.tight_layout()
    plt.savefig(outdir / f"permutation_importance_{set_name}_{model_name}.png", dpi=300)
    plt.close()

    return df_pi


# -------------------------------
# Dataset loaders
# -------------------------------
def load_cdc():
    cdc = fetch_ucirepo(id=891)
    X = cdc.data.features.copy()
    y = cdc.data.targets.to_numpy().ravel()
    y = (y > 0).astype(int)  # binarize
    X = X.dropna(axis=1, how='all')
    X = X.loc[:, X.nunique() > 1]
    return X, y

def load_pima(local_csv=None):
    if local_csv and Path(local_csv).exists():
        df = pd.read_csv(local_csv)
        if 'Outcome' in df.columns:
            y = df['Outcome'].values.astype(int)
            X = df.drop(columns=['Outcome'])
        else:
            y = df.iloc[:, -1].values.astype(int)
            X = df.iloc[:, :-1]
        return X, y
    for pid in [17, 18, 37, 38, 768]:
        try:
            ds = fetch_ucirepo(id=pid)
            meta = str(ds.metadata.name).lower()
            if 'pima' in meta and 'diabetes' in meta:
                X = ds.data.features.copy()
                if hasattr(ds.data, 'targets') and ds.data.targets is not None:
                    y_raw = ds.data.targets
                    if isinstance(y_raw, pd.DataFrame) and y_raw.shape[1] == 1:
                        y = y_raw.iloc[:, 0].to_numpy().ravel().astype(int)
                    else:
                        y = np.array(y_raw).ravel().astype(int)
                else:
                    X = ds.data.original.copy()
                    y = X.iloc[:, -1].values.astype(int)
                    X = X.iloc[:, :-1]
                return X, y
        except Exception:
            continue
    raise RuntimeError("Could not load PIMA dataset. Provide local CSV path to load_pima().")

# -------------------------------
# Core FS runners (Gumbel, MI, L1EN, mRMR, ReliefF)
# -------------------------------
def run_feature_selectors(X_tr, y_tr, cfg, T):
    # Gumbel
    T.start("fs_gumbel")
    df_gumbel = gumbel_lambdaU_feature_scores(X_tr, y_tr, n_jobs=cfg["N_JOBS"])
    T.stop("fs_gumbel")
    top_gumbel = topk_from_df(df_gumbel, k=cfg["TOPK"], col="Lambda_U")

    # MI
    T.start("fs_mi")
    mi = mutual_info_classif(X_tr, y_tr, random_state=cfg["SEED"])
    df_mi = pd.Series(mi, index=X_tr.columns).sort_values(ascending=False)
    top_mi = df_mi.head(cfg["TOPK"]).index.tolist()
    T.stop("fs_mi")

    # L1/EN (elasticnet)
    T.start("fs_l1en")
    top_l1en = select_l1_enet(X_tr, y_tr, k=cfg["TOPK"], seed=cfg["SEED"], penalty='elasticnet', l1_ratio=0.7)
    T.stop("fs_l1en")

    # mRMR (custom)
    T.start("fs_mrmr")
    top_mrmr = mrmr_select(X_tr, y_tr, k=cfg["TOPK"], seed=cfg["SEED"])
    T.stop("fs_mrmr")

    # ReliefF
    T.start("fs_relieff")
    df_relf = relieff_rank_table(X_tr, y_tr, seed=cfg["SEED"], n_neighbors=10, n_samples=None)
    top_relf = df_relf.head(cfg["TOPK"])["Feature"].tolist()
    T.stop("fs_relieff")

    return df_gumbel, top_gumbel, df_mi, top_mi, top_l1en, top_mrmr, df_relf, top_relf



def bootstrap_auc_ci(y_true, y_prob, B=2000, seed=123, alpha=0.05):
    """
    Bootstrap CI for ROC AUC by resampling test indices with replacement.
    Returns (lo, hi). Does not change the point estimate.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    # precompute indices for speed + determinism
    aucs = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        # if resample has only one class, skip by drawing again
        if yt.min() == yt.max():
            # try a few quick redraws
            for _ in range(10):
                idx = rng.integers(0, n, size=n)
                yt = y_true[idx]; yp = y_prob[idx]
                if yt.min() != yt.max():
                    break
        if yt.min() == yt.max():
            aucs[b] = np.nan
        else:
            aucs[b] = roc_auc_score(yt, yp)

    aucs = aucs[~np.isnan(aucs)]
    lo = float(np.quantile(aucs, alpha/2))
    hi = float(np.quantile(aucs, 1 - alpha/2))
    return lo, hi


# Evaluation (models x feature sets)
# -------------------------------
# pass full train (X_trF,y_trF) + split train (X_trA,y_trA) + val + test
def evaluate_feature_sets(dataset_name,
                          X_trF, y_trF,    # full training = X_trA ∪ X_val
                          X_trA, y_trA,    # subset used to learn parameters before thresholding
                          X_val, y_val,
                          X_te,  y_te,
                          feature_sets, cfg, outdir):
    EVAL_MODELS = cfg["EVAL_MODELS"]
    SEED = cfg["SEED"]

    # class balance / weights based on FULL training set
    pos = np.sum(y_trF == 1); neg = np.sum(y_trF == 0)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    results = []
    roc_store_by_model = defaultdict(list)
    pr_store_by_model  = defaultdict(list)
    roc_store_by_set   = defaultdict(list)
    pr_store_by_set    = defaultdict(list)

    for set_name, feats in feature_sets.items():
        # split-train / val / test slices for THIS feature set
        Xtr_set  = X_trA[feats]
        Xval_set = X_val[feats]
        Xte_set  = X_te[feats]

        # scaler for LR only (fit on split-train)
        scaler_lr = StandardScaler().fit(Xtr_set)

        for m_name in EVAL_MODELS:
            model = build_model(m_name, cfg, SEED, scale_pos_weight)

            # inputs for threshold learning
            if m_name == 'LR':
                Xtr_use  = scaler_lr.transform(Xtr_set)
                Xval_use = scaler_lr.transform(Xval_set)
            else:
                Xtr_use, Xval_use = Xtr_set, Xval_set

            # GB uses sample weights; learn parameters on split-train
            sample_w = compute_sample_weight(class_weight='balanced', y=y_trA) if m_name=='GB' else None

            if isinstance(model, CalibratedClassifierCV):
                model.fit(Xtr_use, y_trA, sample_weight=None if m_name!='GB' else sample_w)
            else:
                model.fit(Xtr_use, y_trA, sample_weight=sample_w)

            # choose threshold on validation
            y_val_prob = model.predict_proba(Xval_use)[:, 1]
            thr = pick_threshold(y_val, y_val_prob, criterion=cfg["THRESH_CRITERION"])

            # ---- FINAL FIT on FULL TRAIN (X_trF, y_trF) ----
            model_full = build_model(m_name, cfg, SEED, scale_pos_weight)
            if m_name == 'LR':
                scaler_full  = StandardScaler().fit(X_trF[feats])
                Xtr_full_use = scaler_full.transform(X_trF[feats])
                Xte_full_use = scaler_full.transform(X_te[feats])
            else:
                Xtr_full_use = X_trF[feats]
                Xte_full_use = X_te[feats]

            sample_w_full = compute_sample_weight(class_weight='balanced', y=y_trF) if m_name=='GB' else None

            if isinstance(model_full, CalibratedClassifierCV):
                model_full.fit(Xtr_full_use, y_trF, sample_weight=None if m_name!='GB' else sample_w_full)
            else:
                model_full.fit(Xtr_full_use, y_trF, sample_weight=sample_w_full)

            # TEST metrics (threshold from val)
            y_prob = model_full.predict_proba(Xte_full_use)[:, 1]
            y_pred = (y_prob >= thr).astype(int)

            auc  = roc_auc_score(y_te, y_prob)
            auc_lo, auc_hi = bootstrap_auc_ci(y_te, y_prob, B=1000, seed=cfg["SEED"])
            ap   = average_precision_score(y_te, y_prob)
            bal  = balanced_accuracy_score(y_te, y_pred)
            cr   = classification_report(y_te, y_pred, output_dict=True)

            results.append({
                'Dataset': dataset_name, 'Set': set_name, 'Model': m_name, 'Threshold': thr,
                'Accuracy': cr['accuracy'],
                'Balanced_Acc': bal,
                'Precision': cr['weighted avg']['precision'],
                'Recall': cr['weighted avg']['recall'],
                'F1': cr['weighted avg']['f1-score'],
                'ROC_AUC': auc, 'PR_AUC': ap,
                'ROC_AUC_CI_L': auc_lo, 'ROC_AUC_CI_U': auc_hi
            })

            # Curves & saves
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            prec, rec, _ = precision_recall_curve(y_te, y_prob)
            np.save(outdir / f"probs_{set_name}_{m_name}.npy", y_prob)
            np.save(outdir / f"preds_{set_name}_{m_name}.npy", y_pred)

            # Individual plots
            plt.figure(figsize=(5,4))
            plt.plot(fpr, tpr, label=f"AUC={auc:.3f}"); plt.plot([0,1],[0,1],'--',lw=1)
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC • {dataset_name} • {set_name} • {m_name}")
            plt.legend(loc="lower right"); plt.tight_layout()
            plt.savefig(outdir / f"roc_{set_name}_{m_name}.png", dpi=300); plt.close()

            plt.figure(figsize=(5,4))
            plt.plot(rec, prec, label=f"AP={ap:.3f}")
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title(f"PR • {dataset_name} • {set_name} • {m_name}")
            plt.legend(loc="lower left"); plt.tight_layout()
            plt.savefig(outdir / f"pr_{set_name}_{m_name}.png", dpi=300); plt.close()

            plot_confmat(set_name, m_name, y_te, y_pred, outdir)

            # For superimposed plots
            roc_store_by_model[m_name].append((set_name, fpr, tpr, auc))
            pr_store_by_model[m_name].append((set_name, rec, prec, ap))
            roc_store_by_set[set_name].append((m_name, fpr, tpr, auc))
            pr_store_by_set[set_name].append((m_name, rec, prec, ap))

    df_res = pd.DataFrame(results).sort_values(['Set','Model']).reset_index(drop=True)
    df_res.to_csv(outdir / "feature_selection_comparison_results.csv", index=False)

    # superimposed per-model (sets)
    for m_name in set(df_res['Model']):
        if m_name in roc_store_by_model:
            plot_superimposed_roc_by_model(m_name, roc_store_by_model[m_name], outdir)
        if m_name in pr_store_by_model:
            plot_superimposed_pr_by_model(m_name, pr_store_by_model[m_name], outdir)

    # superimposed per-set (models)
    for s_name in set(df_res['Set']):
        if s_name in roc_store_by_set:
            plot_superimposed_roc_by_set(s_name, roc_store_by_set[s_name], outdir)
        if s_name in pr_store_by_set:
            plot_superimposed_pr_by_set(s_name, pr_store_by_set[s_name], outdir)

    # metric curves
    METRICS = ['Accuracy','Balanced_Acc','ROC_AUC','PR_AUC','Precision','Recall','F1']
    core = ['All','Gumbel','MI','L1EN','mRMR','ReliefF']
    extras = [s for s in df_res['Set'].unique() if s not in core]
    order = [s for s in core if s in df_res['Set'].unique()] + extras

    for m in set(df_res['Model']):
        for metric in METRICS:
            plot_metric_curve_for_model(df_res, m, metric, outdir, order=order)
    for metric in METRICS:
        plot_metric_curve_all_models(df_res, metric, outdir, order=order, models=sorted(set(df_res['Model'])))

    return df_res


# -------------------------------
# Stability (Jaccard / rank corr)
# -------------------------------
def selection_stability(Xtr, ytr, k=5, B=20, seed=123, outdir=None, all_cols=None):
    rng = np.random.default_rng(seed); n = len(ytr)
    sets, ranks = [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        Xb, yb = Xtr.iloc[idx], ytr[idx]
        dfb = gumbel_lambdaU_feature_scores(Xb, yb, n_jobs=1)
        sets.append(set(dfb.nlargest(k, "Lambda_U")["Feature"].tolist()))
        ranks.append(dfb.set_index("Feature")["Lambda_U"].rank(ascending=False, method="average"))
    jaccs = []
    for i in range(B-1):
        for j in range(i+1, B):
            a, b = sets[i], sets[j]
            jaccs.append(len(a & b) / max(1, len(a | b)))
    anchor = ranks[0].reindex(all_cols)
    rhos = []
    for r in ranks[1:]:
        r2 = r.reindex(all_cols)
        rho, _ = spearmanr(anchor, r2, nan_policy="omit")
        rhos.append(rho)
    if outdir:
        pd.DataFrame({"Jaccard": jaccs}).to_csv(outdir / "selection_stability_jaccard.csv", index=False)
        pd.DataFrame({"Spearman_rho": rhos}).to_csv(outdir / "selection_stability_rankcorr.csv", index=False)
    return float(np.mean(jaccs)), float(np.std(jaccs)), float(np.nanmean(rhos)), float(np.nanstd(rhos))

# -------------------------------
# k-sweep
# -------------------------------
def gumbel_k_sweep(df_gumbel, X_tr, X_val, X_te, y_tr, y_val, y_te, cfg, outdir, dataset_name):
    rows = []
    for k in cfg["K_SWEEP"]:
        top_k = topk_from_df(df_gumbel, k=k)
        for m_name in cfg["EVAL_MODELS"]:
            scaler_lr = StandardScaler().fit(X_tr[top_k])
            model = build_model(m_name, cfg, cfg["SEED"], (np.sum(y_tr==0)/max(1,np.sum(y_tr==1))))
            if m_name == 'LR':
                Xtr_use = scaler_lr.transform(X_tr[top_k])
                Xval_use = scaler_lr.transform(X_val[top_k])
                Xte_use  = scaler_lr.transform(X_te[top_k])
            else:
                Xtr_use, Xval_use, Xte_use = X_tr[top_k], X_val[top_k], X_te[top_k]
            sw = compute_sample_weight(class_weight='balanced', y=y_tr) if m_name=='GB' else None
            model.fit(Xtr_use, y_tr, sample_weight=sw)
            thr = pick_threshold(y_val, model.predict_proba(Xval_use)[:,1], criterion=cfg["THRESH_CRITERION"])
            model_full = build_model(m_name, cfg, cfg["SEED"], (np.sum(y_tr==0)/max(1,np.sum(y_tr==1))))
            if m_name == 'LR':
                scf = StandardScaler().fit(X_tr[top_k])
                Xtr_fu = scf.transform(X_tr[top_k]); Xte_fu = scf.transform(X_te[top_k])
            else:
                Xtr_fu, Xte_fu = X_tr[top_k], X_te[top_k]
            model_full.fit(Xtr_fu, y_tr, sample_weight=(compute_sample_weight(class_weight='balanced', y=y_tr) if m_name=='GB' else None))
            probs = model_full.predict_proba(Xte_fu)[:,1]
            rows.append({
                "Dataset": dataset_name, "k": k, "Model": m_name,
                "ROC_AUC": roc_auc_score(y_te, probs),
                "PR_AUC": average_precision_score(y_te, probs)
            })
    dfk = pd.DataFrame(rows)
    dfk.to_csv(outdir / "gumbel_k_sweep.csv", index=False)
    return dfk

# -------------------------------
# Perturbation tests (light)
# -------------------------------
def eval_under_perturbations(Xtr, ytr, Xte, yte, feats, cfg, model_name='GB'):
    rng = np.random.default_rng(cfg["SEED"])
    rows = []

    def run_once(Xtr_p, ytr_p, Xte_p, tag, level):
        model = build_model(model_name, cfg, cfg["SEED"], (np.sum(ytr_p==0)/max(1,np.sum(ytr_p==1))))
        sw = compute_sample_weight(class_weight='balanced', y=ytr_p) if model_name=='GB' else None
        model.fit(Xtr_p, ytr_p, sample_weight=sw)
        yprob = model.predict_proba(Xte_p)[:,1]
        return {"Type": tag, "Level": level,
                "ROC_AUC": roc_auc_score(yte, yprob),
                "PR_AUC": average_precision_score(yte, yprob)}

    rows.append(run_once(Xtr[feats], ytr, Xte[feats], "baseline", 0.0))

    for pf in cfg["ROB_LABEL_FLIP"]:
        if pf <= 0:
            continue
        ytr_p = ytr.copy()
        flip_idx = rng.choice(len(ytr_p), size=int(pf*len(ytr_p)), replace=False)
        ytr_p[flip_idx] = 1 - ytr_p[flip_idx]
        rows.append(run_once(Xtr[feats], ytr_p, Xte[feats], "label_flip", pf))

    for sd in cfg["ROB_NOISE_SD"]:
        if sd <= 0:
            continue
        Xtr_p = Xtr[feats].copy(); Xte_p = Xte[feats].copy()
        for c in feats:
            if np.issubdtype(Xtr_p[c].dtype, np.number):
                s = Xtr_p[c].std(ddof=0)
                Xtr_p[c] += rng.normal(0, sd*s, size=len(Xtr_p))
                Xte_p[c] += rng.normal(0, sd*s, size=len(Xte_p))
        rows.append(run_once(Xtr_p, ytr, Xte_p, "feat_noise", sd))

    for mc in cfg["ROB_MCAR"]:
        if mc <= 0:
            continue
        Xtr_p = Xtr[feats].copy(); Xte_p = Xte[feats].copy()
        for c in feats:
            miss_tr = rng.choice(len(Xtr_p), size=int(mc*len(Xtr_p)), replace=False)
            miss_te = rng.choice(len(Xte_p), size=int(mc*len(Xte_p)), replace=False)
            Xtr_p.loc[Xtr_p.index[miss_tr], c] = np.nan
            Xte_p.loc[Xte_p.index[miss_te], c] = np.nan
            med = np.nanmedian(Xtr_p[c].values)
            Xtr_p[c] = Xtr_p[c].fillna(med); Xte_p[c] = Xte_p[c].fillna(med)
        rows.append(run_once(Xtr_p, ytr, Xte_p, "mcar", mc))

    for ss in cfg["ROB_SUBSAMPLE"]:
        if ss >= 1.0:
            continue
        keep = rng.choice(len(Xtr), size=int(ss*len(Xtr)), replace=False)
        Xtr_p, ytr_p = Xtr.iloc[keep][feats], ytr[keep]
        rows.append(run_once(Xtr_p, ytr_p, Xte[feats], "subsample", ss))

    return pd.DataFrame(rows)

def compute_permutation_importance_for_gumbel_best(
    X_tr, y_tr, X_te, y_te, feats, model_name, cfg, outdir, n_repeats=500
):
    """
    Trains the requested model on TRAIN using only `feats` (Gumbel top-k),
    then computes test-set permutation importance (ROC_AUC scorer).
    Saves CSV + a bar plot.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # build model consistent with the rest of the code (calibration, seeds, weights)
    pos = np.sum(y_tr == 1); neg = np.sum(y_tr == 0)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    model = build_model(model_name, cfg, cfg["SEED"], scale_pos_weight)

    # LR uses standardization like everywhere else
    if model_name == 'LR':
        scaler = StandardScaler().fit(X_tr[feats])
        Xtr_use = scaler.transform(X_tr[feats])
        Xte_use = scaler.transform(X_te[feats])
    else:
        Xtr_use = X_tr[feats]
        Xte_use = X_te[feats]

    # GB can use sample weights (consistent with your training)
    sample_w = compute_sample_weight(class_weight='balanced', y=y_tr) if model_name=='GB' else None

    # fit
    if isinstance(model, CalibratedClassifierCV):
        model.fit(Xtr_use, y_tr, sample_weight=None if model_name!='GB' else sample_w)
    else:
        model.fit(Xtr_use, y_tr, sample_weight=sample_w)

    # permutation importance on held-out TEST with ROC_AUC scorer
    pi = permutation_importance(
        model, Xte_use, y_te,
        scoring='roc_auc',
        n_repeats=n_repeats,
        random_state=cfg["SEED"],
        n_jobs=cfg["N_JOBS"]
    )

    df_pi = pd.DataFrame({
        "Feature": feats,
        "Importance_Mean": pi.importances_mean,
        "Importance_STD":  pi.importances_std
    }).sort_values("Importance_Mean", ascending=False)

    # save CSV
    csv_path = outdir / f"permutation_importance_Gumbel_{model_name}.csv"
    df_pi.to_csv(csv_path, index=False)

    # bar plot
    plt.figure(figsize=(7.2, 4.5))
    order = df_pi["Feature"].values
    means = df_pi["Importance_Mean"].values
    stds  = df_pi["Importance_STD"].values
    plt.bar(range(len(order)), means, yerr=stds, capsize=3)
    plt.xticks(range(len(order)), order, rotation=30, ha='right')
    plt.ylabel("Permutation importance (Δ ROC AUC)")
    plt.title(f"Gumbel top-k • {model_name} • n_repeats={n_repeats}")
    plt.tight_layout()
    png_path = outdir / f"permutation_importance_Gumbel_{model_name}.png"
    plt.savefig(png_path, dpi=300); plt.close()

    return str(csv_path), str(png_path)


# -------------------------------
# Minimal DeLong implementation (paired AUC)
# -------------------------------
# --- drop-in DeLong (covariance-aware) replacement ---

import numpy as np
from scipy.stats import norm

def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out

def _delong_covariance(preds, y_true):
    """
    preds: array (k_models, n_samples) of scores
    y_true: array (n_samples,) of {0,1}
    Returns: aucs (k,), covariance matrix S (k,k) for AUCs (DeLong)
    """
    y_true = np.asarray(y_true).astype(int)
    preds = np.asarray(preds)
    pos = (y_true == 1)
    m = int(pos.sum())
    n = preds.shape[1] - m
    if m == 0 or n == 0:
        raise ValueError("DeLong requires at least one positive and one negative sample.")

    # sort so that positives come first
    order = np.argsort(~pos)            # positives first (False=0 sorted before True=1)
    P = preds[:, order]

    # midranks
    k = P.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(P[r, :m])
        ty[r] = _midrank(P[r, m:])
        tz[r] = _midrank(P[r])

    # AUCs
    aucs = (tz[:, :m].sum(axis=1) - m*(m+1)/2.0) / (m*n)

    # DeLong components (each row r is a model; columns are observations)
    V01 = (tz[:, :m] - tx) / n          # per-positive contributions
    V10 = 1.0 - (tz[:, m:] - ty) / m    # per-negative contributions

    # Empirical covariance across models
    # np.cov expects variables in rows, observations in columns (rowvar=True)
    S01 = np.cov(V01, bias=False) if m > 1 else np.zeros((k, k))
    S10 = np.cov(V10, bias=False) if n > 1 else np.zeros((k, k))

    # DeLong covariance of AUCs
    S = S01 / m + S10 / n
    return aucs, S

def delong_roc_test(y_true, p1, p2):
    preds = np.vstack([p1, p2])
    aucs, S = _delong_covariance(preds, y_true)
    diff = aucs[0] - aucs[1]
    var = S[0,0] + S[1,1] - 2.0*S[0,1]   # includes covariance term
    z = diff / np.sqrt(max(var, 1e-12))
    return float(2.0 * (1.0 - norm.cdf(abs(z))))



def run_significance_tests(y_te, outdir, anchor_set, models=("RF","XGB","LR","GB")):
    outdir = Path(outdir)
    rows_delong, rows_mcn = [], []
    for m in models:
        files = glob.glob(str(outdir / f"probs_*_{m}.npy"))
        sets = []
        for fp in files:
            mobj = re.search(r"probs_(.+?)_" + re.escape(m) + r"\.npy$", fp)
            if mobj: sets.append(mobj.group(1))
        sets = sorted(set(sets))
        if anchor_set not in sets: continue
        pa = np.load(outdir / f"probs_{anchor_set}_{m}.npy")
        for other in sets:
            if other == anchor_set: continue
            pb = np.load(outdir / f"probs_{other}_{m}.npy")
            pval = float(delong_roc_test(y_te, pa, pb))
            rows_delong.append({"Model": m, "A": anchor_set, "B": other, "DeLong_p": pval})
        # McNemar on correctness
        ya = np.load(outdir / f"preds_{anchor_set}_{m}.npy")
        ca = (ya == y_te).astype(int)
        for other in sets:
            if other == anchor_set: continue
            yb = np.load(outdir / f"preds_{other}_{m}.npy")
            cb = (yb == y_te).astype(int)
            M = confusion_matrix(ca, cb, labels=[0, 1])
            res = mcnemar(M, exact=False, correction=True)
            rows_mcn.append({"Model": m, "A": anchor_set, "B": other,
                             "McNemar_stat": float(res.statistic), "McNemar_p": float(res.pvalue)})
    pd.DataFrame(rows_delong).to_csv(outdir / "significance_delong.csv", index=False)
    pd.DataFrame(rows_mcn).to_csv(outdir / "significance_mcnemar.csv", index=False)

# -------------------------------
# Main experiment runner for one dataset
# -------------------------------
def run_experiment(dataset_name, X, y, cfg=None, out_root="./outputs"):
    cfg = {**CFG, **(cfg or {})}
    set_seed(cfg["SEED"])

    outdir = Path(out_root) / dataset_name
    outdir.mkdir(parents=True, exist_ok=True) 
    figdir = outdir / "figures"; figdir.mkdir(parents=True, exist_ok=True)

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg["TEST_SIZE"],
                                              random_state=cfg["SEED"], stratify=y)

    # --- PIMA hygiene: treat zeros as missing + median impute (fit on TRAIN only) ---
    if dataset_name.upper() == "PIMA":
        X_tr, X_te = pima_zero_impute_train_only(
            X_tr=X_tr,
            X_te=X_te,
            outdir=outdir,
            cols=cfg.get("PIMA_ZERO_MISS_COLS", []),
            enable=cfg.get("PIMA_TREAT_ZEROS_AS_MISSING", True),
        )


    # Save indices + config
    np.save(outdir / "splits_train_idx.npy", np.arange(len(y_tr)))
    np.save(outdir / "splits_test_idx.npy",  np.arange(len(y_te)))
    with open(outdir / "config.json","w") as f: json.dump(cfg, f, indent=2)

    # Class balance
    pd.DataFrame([class_imbalance_report(y, "full"),
                  class_imbalance_report(y_tr, "train"),
                  class_imbalance_report(y_te, "test")]).to_csv(outdir / "class_balance.csv", index=False)

    # Train/val from train
    X_trA, X_val, y_trA, y_val = train_test_split(X_tr, y_tr, test_size=cfg["VAL_SIZE"],
                                                  random_state=cfg["SEED"], stratify=y_tr)

    # Timings
    T = Timer()
    T.start("total")

    # FS on TRAIN only
    df_gumbel, top_gumbel, df_mi, top_mi, top_l1en, top_mrmr, df_relf, top_relf = run_feature_selectors(X_tr, y_tr, cfg, T)
    df_gumbel.to_csv(outdir / "gumbel_feature_scores.csv", index=False)
    pd.DataFrame({"Feature": X_tr.columns,
                  "Score": mutual_info_classif(X_tr, y_tr, random_state=cfg["SEED"])}).sort_values("Score", ascending=False)\
        .to_csv(outdir / "ranking_mi.csv", index=False)
    rank_l1_enet_table(X_tr, y_tr, seed=cfg["SEED"], penalty='elasticnet', l1_ratio=0.7)\
        .to_csv(outdir / "ranking_l1en.csv", index=False)
    mrmr_rank_table(X_tr, y_tr, seed=cfg["SEED"]).to_csv(outdir / "ranking_mrmr.csv", index=False)
    df_relf.to_csv(outdir / "ranking_relieff.csv", index=False)

    # λU CIs bar (B=1000)
    if cfg["BOOTSTRAP_LAM"] and len(top_gumbel) > 0:
        feats = top_gumbel
        means, lo, hi = bootstrap_lambdaU_ci(X_tr, y_tr, feats, B=cfg["N_BOOT_LAM"], seed=cfg["SEED"], n_jobs=cfg["N_JOBS"])
        order_idx = np.argsort(means)[::-1]
        feats_sorted = [feats[i] for i in order_idx]
        means_sorted, lo_sorted, hi_sorted = means[order_idx], lo[order_idx], hi[order_idx]
        plt.figure(figsize=(7,4))
        plt.bar(range(len(feats_sorted)), means_sorted,
                yerr=[means_sorted - lo_sorted, hi_sorted - means_sorted], capsize=3)
        plt.xticks(range(len(feats_sorted)), feats_sorted, rotation=30, ha='right')
        plt.ylabel(r"Gumbel $\lambda_U$"); plt.title(f"{dataset_name}: Top-{len(feats_sorted)} λU (95% CI, B={cfg['N_BOOT_LAM']})")
        plt.tight_layout(); plt.savefig(figdir / "lambdaU_bar_topk.png", dpi=300); plt.close()

    # Master feature sets
    feature_sets = {
        'All':     X_tr.columns.tolist(),
        'Gumbel':  top_gumbel,
        'MI':      top_mi,
        'L1EN':    top_l1en,
        'mRMR':    top_mrmr,
        'ReliefF': top_relf,
    }

    # ---- Save top-k feature summary (for paper/thesis) ----
    pd.DataFrame({
        "Method":   ["Gumbel", "MI", "L1EN", "mRMR", "ReliefF"],
        "TopK":     [cfg["TOPK"]]*5,
        "Features": [
            ", ".join(feature_sets["Gumbel"]),
            ", ".join(feature_sets["MI"]),
            ", ".join(feature_sets["L1EN"]),
            ", ".join(feature_sets["mRMR"]),
            ", ".join(feature_sets["ReliefF"]),
        ],
    }).to_csv(outdir / "topk_features_summary.csv", index=False)

    # Evaluate all sets/models
    df_res = evaluate_feature_sets(dataset_name,
                               X_tr,  y_tr,     # full train
                               X_trA, y_trA,    # subset train
                               X_val, y_val,
                               X_te,  y_te,
                               feature_sets, cfg, figdir)

    df_res.to_csv(outdir / "feature_selection_comparison_results.csv", index=False)

    # Significance tests anchored on Gumbel with best model
    PRIMARY_METRIC = cfg.get("PRIMARY_METRIC", "ROC_AUC")
    best_row   = df_res.sort_values(PRIMARY_METRIC, ascending=False).iloc[0]
    best_set   = best_row["Set"]; best_model = best_row["Model"]
    print(f"[{dataset_name}] Best by {PRIMARY_METRIC}: Set={best_set}, Model={best_model}")
    run_significance_tests(y_te, figdir, anchor_set="Gumbel", models=(best_model,))

    # ---- Permutation importance for the best (set, model) on the TEST set ----
    best_feats = feature_sets[best_set]
    _ = compute_permutation_importance_for_best(
            X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
            feats=best_feats, model_name=best_model, cfg=cfg,
            outdir=figdir, set_name=best_set, n_repeats=500)

    # --- Permutation importance for Gumbel top-k using the best model within Gumbel ---
    if "Gumbel" in feature_sets and len(feature_sets["Gumbel"]) > 0:
        # choose the best model *within* the Gumbel set by the primary metric
        sub_g = df_res[df_res["Set"] == "Gumbel"]
        if not sub_g.empty:
            best_model_gumbel = sub_g.sort_values(PRIMARY_METRIC, ascending=False).iloc[0]["Model"]
            g_feats = feature_sets["Gumbel"]
            csv_pi, png_pi = compute_permutation_importance_for_gumbel_best(
                X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
                feats=g_feats, model_name=best_model_gumbel, cfg=cfg,
                outdir=figdir, n_repeats=500  # adjust to 100/200/500/1000 as you prefer
            )
            print(f"[{dataset_name}] Permutation importance (Gumbel, {best_model_gumbel}) -> {csv_pi}")



    # k-sweep
    if cfg["DO_K_SWEEP"]:
        dfk = gumbel_k_sweep(df_gumbel, X_trA, X_val, X_te, y_trA, y_val, y_te, cfg, outdir, dataset_name)
        print(f"[{dataset_name}] k-sweep saved -> {outdir/'gumbel_k_sweep.csv'}")

    # Stability
    if cfg["STABILITY"]:
        j_mu, j_sd, r_mu, r_sd = selection_stability(X_tr, y_tr, k=cfg["TOPK"], B=cfg["STABILITY_B"],
                                                     seed=cfg["SEED"], outdir=outdir, all_cols=X_tr.columns)
        with open(outdir / "selection_stability_summary.txt","w") as f:
            f.write(f"Jaccard mean±sd: {j_mu:.3f}±{j_sd:.3f}\n")
            f.write(f"Spearman rho mean±sd: {r_mu:.3f}±{r_sd:.3f}\n")


    # Perturbations (GB for CDC; RF for PIMA)
    if cfg["PERTURB"] and len(top_gumbel) > 0:
        rob_model = 'GB' if dataset_name.upper() == 'CDC' else 'RF'
        rob = eval_under_perturbations(X_tr, y_tr, X_te, y_te, top_gumbel, cfg, model_name=rob_model)
        rob.to_csv(outdir / "robustness_summary.csv", index=False)


    # Timing & notes
    T.stop("total")
    T.dump(outdir / "runtime_profile.csv")
    with open(outdir / "complexity_note.txt","w") as f:
        n, d = X.shape[0], X.shape[1]
        f.write("Complexity notes (informal):\n")
        f.write(f"n={n}, d={d}\n")
        f.write("- Gumbel λU via Kendall's τ (method-of-moments): per-feature dominated by τ ~ O(n log n) in practice; overall ≈ O(d·n log n).\n")
        f.write("- MI: standard filter; sklearn mutual_info_classif (approx). mRMR: greedy MI−corr proxy.\n")
        f.write("- L1/ElasticNet: embedded; cost of logistic training with SAGA on standardized X.\n")
        f.write("- ReliefF: k-NN (brute) per sampled row; memory O(n·d), practical with float32 on CDC/PIMA.\n")

    print(f"[{dataset_name}] Done. Outputs in: {outdir.resolve()}")
    return str(outdir.resolve())

# -------------------------------
# Entrypoint (CDC + PIMA)
# -------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pima_csv", type=str, default="", help="Optional path to local PIMA CSV (Outcome column as label).")
    p.add_argument("--jobs", type=int, default=CFG["N_JOBS"], help="Parallel jobs for λU/bootstraps (1-2 recommended).")
    p.add_argument("--seed", type=int, default=CFG["SEED"])
    p.add_argument("--out", type=str, default=CFG["OUT_ROOT"])
    args = p.parse_args()

    CFG["N_JOBS"] = int(args.jobs)
    CFG["SEED"] = int(args.seed)

    # CDC config
    cfg_cdc = {
        "TOPK": 7,
        "K_SWEEP": [3,5,7,10],
        "DO_K_SWEEP": True
    }
    # PIMA config
    cfg_pima = {
        "TOPK": 8,
        "K_SWEEP": [3,5,8],
        "DO_K_SWEEP": True
    }

    # CDC
    X_cdc, y_cdc = load_cdc()
    out_dir_cdc = run_experiment("CDC", X_cdc, y_cdc, cfg=cfg_cdc, out_root=args.out)
    print("CDC outputs ->", out_dir_cdc)

    # PIMA
    pima_path = "/content/pima.csv"
    X_pim, y_pim = load_pima(pima_path)
    out_dir_pima = run_experiment("PIMA", X_pim, y_pim, cfg=cfg_pima, out_root=args.out)
    print("PIMA outputs ->", out_dir_pima)

if __name__ == "__main__":
    # Override config for Colab (no argparse)
    CFG["N_JOBS"] = 2
    CFG["SEED"] = 123
    CFG["OUT_ROOT"] = "./outputs"

    # CDC config
    cfg_cdc = {
        "TOPK": 10,
        "K_SWEEP": [3, 5, 7, 10],
        "DO_K_SWEEP": True
    }

    # PIMA config
    cfg_pima = {
        "TOPK": 8,
        "K_SWEEP": [3, 5, 8],
        "DO_K_SWEEP": True
    }

    # ---- Run CDC ----
    X_cdc, y_cdc = load_cdc()
    out_dir_cdc = run_experiment("CDC", X_cdc, y_cdc, cfg=cfg_cdc, out_root=CFG["OUT_ROOT"])
    print("CDC outputs ->", out_dir_cdc)

    # ---- Run PIMA ----
    pima_path = "/content/pima.csv"   # 👈 Your file path! I have used Google Colab Pro!
    X_pim, y_pim = load_pima(pima_path)
    out_dir_pima = run_experiment("PIMA", X_pim, y_pim, cfg=cfg_pima, out_root=CFG["OUT_ROOT"])
    print("PIMA outputs ->", out_dir_pima)
