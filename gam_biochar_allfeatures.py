# gam_biochar_allfeatures.py
import argparse, json, math, os, sys, warnings, pickle, itertools
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures to files
import matplotlib.pyplot as plt
import argparse, json, math, os, sys, warnings, pickle, itertools
warnings.filterwarnings("ignore", category=UserWarning)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

try:
    from pygam import LinearGAM, s, te
except Exception:
    print("Missing dependency: pygam. Install with: python -m pip install pygam")
    sys.exit(1)

try:
    import shap
except Exception:
    print("Missing dependency: shap. Install with: python -m pip install shap")
    sys.exit(1)
# ---------------- helpers ----------------
def winsorize(x: pd.Series, q=0.01):
    lo, hi = x.quantile(q), x.quantile(1 - q)
    return x.clip(lo, hi)

def flatten_lam(lam):
    try:
        return np.array(lam, dtype=float).ravel().tolist()
    except Exception:
        out, stack = [], [lam]
        while stack:
            v = stack.pop()
            if isinstance(v, (list, tuple, np.ndarray)):
                stack.extend(list(v))
            else:
                try: out.append(float(v))
                except Exception: pass
        return out

def sanitize(name: str) -> str:
    bad = [' ', '/', '\\', '(', ')', '%', ':', ';', ',', '[', ']', '{', '}', '|', '"', "'"]
    for b in bad: name = name.replace(b, "_")
    while "__" in name: name = name.replace("__", "_")
    return name.strip("_")

def savefig(fname):
    plt.tight_layout(); plt.savefig(fname, dpi=180); plt.close()
    print("Saved:", os.path.abspath(fname))

def plot_parity(y_true, y_pred, fname="cv_parity.png", title="Actual vs Predicted (OOF)"):
    plt.figure(figsize=(6.5,6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="black", linewidths=0.5)
    mn, mx = np.min([y_true,y_pred]), np.max([y_true,y_pred])
    pad = 0.05*(mx-mn + 1e-9); a, b = mn-pad, mx+pad
    plt.plot([a,b],[a,b],"--")
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    savefig(fname)

def plot_permutation_importance(df, fname="importance_permutation_top10.png"):
    plt.figure(figsize=(8, 6))
    top10 = df.head(10)
    features = top10["feature"][::-1]
    importance = top10["perm_drop_R2"][::-1]
    plt.barh(features, importance, edgecolor="black", linewidth=0.7)
    plt.xlabel("Permutation Importance (Drop in R²)")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importance")
    plt.grid(True, linestyle="--", alpha=0.6, axis='x')
    savefig(fname) 

def pdp_effect_sizes(gam, data, feature_cols, target_col):
    rows = []
    y_mean = data[target_col].mean()
    for i, name in enumerate(feature_cols):
        grid = gam.generate_X_grid(term=i)
        pdp = gam.partial_dependence(term=i, X=grid)
        rows.append({
            "feature": name,
            "pdp_range": float(np.ptp(pdp)),
            "pdp_std": float(np.std(pdp)),
            "pdp_iqr": float(np.percentile(pdp,75)-np.percentile(pdp,25))
        })
        
        plt.figure(figsize=(6,4.5))
        plt.scatter(data[name], data[target_col] - y_mean, 
                    alpha=0.25, color='gray', label='Centered Actual Data')
        plt.plot(grid[:,i], pdp, linewidth=3, label='Partial Dependence (PDP)')
        plt.title(f"Partial Dependence of {name}")
        plt.xlabel(name); plt.ylabel("Effect on Prediction (Centered)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        savefig(f"pdp_{sanitize(name)}.png")
        
    df = pd.DataFrame(rows).sort_values("pdp_range", ascending=False)
    df.to_csv("importance_pdp.csv", index=False)
    print("Saved:", os.path.abspath("importance_pdp.csv"))
    return df

def permutation_importance_cv(X, y, feature_cols, seed, k=5):
    lam_grid = np.logspace(-4, 5, 10)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    drops = np.zeros(len(feature_cols)); counts = np.zeros(len(feature_cols))
    rng = np.random.default_rng(seed)

    for tr, va in kf.split(X,y):
        # additive base + pairwise interactions among top features will be added later in main
        terms = terms = s(0, n_splines=40)
        for i in range(1, X.shape[1]): terms += s(i, n_splines=40)
        gam_cv = LinearGAM(terms).gridsearch(X[tr], y[tr], lam=lam_grid)

        base_r2 = r2_score(y[va], gam_cv.predict(X[va]))
        for j in range(len(feature_cols)):
            Xp = X[va].copy()
            rng.shuffle(Xp[:,j])
            r2p = r2_score(y[va], gam_cv.predict(Xp))
            drops[j] += max(0.0, base_r2 - r2p); counts[j] += 1

    imp = drops/np.maximum(counts,1.0)
    df = pd.DataFrame({"feature": feature_cols, "perm_drop_R2": imp})
    df = df.sort_values("perm_drop_R2", ascending=False)
    df.to_csv("importance_permutation.csv", index=False)
    print("Saved:", os.path.abspath("importance_permutation.csv"))
    return df

def analyze_with_shap(model, df_features, fname="shap_summary_plot.png"):
    """Runs SHAP analysis and saves a summary plot."""
    print("\nRunning SHAP analysis (this may take a moment)...")
    explainer = shap.Explainer(model.predict, df_features)
    shap_values = explainer(df_features)
    plt.figure()
    shap.summary_plot(shap_values, df_features, show=False)
    plt.tight_layout()
    plt.savefig(fname, dpi=180)
    plt.close()
    print("Saved SHAP plot:", os.path.abspath(fname))

def interaction_h_stat(gam, data, feature_cols, pairs, gridN=30):
    """Friedman's H-statistic for interaction strength of pairs."""
    rows=[]
    med = {c: np.median(data[c]) for c in feature_cols}
    for a,b in pairs:
        xa = np.linspace(data[a].quantile(0.02), data[a].quantile(0.98), gridN)
        xb = np.linspace(data[b].quantile(0.02), data[b].quantile(0.98), gridN)
        AA, BB = np.meshgrid(xa, xb); N=AA.size

        # build dataset at medians
        feat = {c: np.full(N, med[c]) for c in feature_cols}
        feat[a] = AA.ravel(); feat[b] = BB.ravel()
        M = np.column_stack([feat[c] for c in feature_cols])

        fij = gam.predict(M).reshape(AA.shape)

        # 1D partials on same grids
        Xa = np.column_stack([feat[c] for c in feature_cols]); Xa[:,feature_cols.index(b)] = med[b]
        fa = gam.predict(Xa).reshape(AA.shape)
        Xb = np.column_stack([feat[c] for c in feature_cols]); Xb[:,feature_cols.index(a)] = med[a]
        fb = gam.predict(Xb).reshape(AA.shape)

        # center surfaces
        fij_c = fij - np.mean(fij)
        fa_c  = fa  - np.mean(fa)
        fb_c  = fb  - np.mean(fb)

        numer = np.var(fij_c - (fa_c + fb_c))
        denom = np.var(fij_c) + 1e-12
        H = np.sqrt(max(0.0, min(1.0, numer/denom)))
        rows.append({"pair": f"{a}__{b}", "feat_a": a, "feat_b": b, "H": float(H)})
    df = pd.DataFrame(rows).sort_values("H", ascending=False)
    df.to_csv("interaction_pairs.csv", index=False)
    print("Saved:", os.path.abspath("interaction_pairs.csv"))
    return df

def plot_surface3d(gam, data, feature_cols, x_name, y_name, Nx=60, Ny=60, x_label=None, y_label=None):
    med = {c: np.median(data[c]) for c in feature_cols}
    xv = np.linspace(data[x_name].quantile(0.02), data[x_name].quantile(0.98), Nx)
    yv = np.linspace(data[y_name].quantile(0.02), data[y_name].quantile(0.98), Ny)
    XX, YY = np.meshgrid(xv, yv); N = XX.size

    feat = {c: np.full(N, med[c]) for c in feature_cols}
    feat[x_name] = XX.ravel(); feat[y_name] = YY.ravel()
    M = np.column_stack([feat[c] for c in feature_cols])

    Z = gam.predict(M).reshape(XX.shape)
    fig = plt.figure(figsize=(7.5,6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(XX, YY, Z, linewidth=0, antialiased=True, cmap="viridis")
    
    # --- บรรทัดที่แก้ไข ---
    ax.set_xlabel(x_label if x_label else x_name)
    ax.set_ylabel(y_label if y_label else y_name)
    # --------------------

    ax.set_zlabel("Predicted Yield-char (%)")
    ax.set_title(f"{x_name} vs {y_name} (others at median)")
    fig.colorbar(surf, shrink=0.65, aspect=10, label="Predicted Yield-char (%)")
    fname = f"surface3d_{sanitize(x_name)}_vs_{sanitize(y_name)}.png"
    savefig(fname)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="Data_raw_CSV.csv")
    ap.add_argument("--target", default="Yield-char")
    ap.add_argument("--winsor", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_search", type=int, default=50000)
    ap.add_argument("--top_k_for_interaction", type=int, default=5,
                    help="build te() for all pairs among top-K features by |Spearman with target|")
    ap.add_argument("--n_pairs_to_plot", type=int, default=3,
                    help="plot 3D surfaces for top-N interacting pairs by H-statistic")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}"); sys.exit(1)

    df = pd.read_csv(args.csv)
    print("Loaded:", df.shape)

    # numeric features
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if args.target not in num_cols:
        print(f"Target column '{args.target}' not found or not numeric."); sys.exit(1)

    feature_cols = [c for c in num_cols if c != args.target]
    data = df[[args.target] + feature_cols].copy()
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    print("Rows kept:", len(data))

    if args.winsor > 0:
        for c in [args.target] + feature_cols:
            data[c] = winsorize(data[c], q=args.winsor)
        print(f"Winsorized at [{args.winsor:.2f}, {1-args.winsor:.2f}]")

    X = data[feature_cols].values
    y = data[args.target].values

    # --- choose pairs for explicit te() by target association ---
    rho = []
    for c in feature_cols:
        try:
            r, _ = spearmanr(data[c], y)
        except Exception:
            r = 0.0
        rho.append((c, abs(r)))
    rho = sorted(rho, key=lambda x: x[1], reverse=True)
    top_feats = [c for c,_ in rho[:max(2, min(args.top_k_for_interaction, len(feature_cols)))]]
    te_pairs = list(itertools.combinations(range(len(feature_cols)), 2))
    te_pairs = [(i,j) for i,j in te_pairs if feature_cols[i] in top_feats and feature_cols[j] in top_feats]
    print("Top features for interactions:", top_feats)
    print("Number of te() pairs:", len(te_pairs))

    # ---------- K-fold CV with inner tuning ----------
    lam_grid = np.logspace(-3,4,8)
    kf = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    oof_pred = np.zeros_like(y, dtype=float); fold_stats=[]

    for fold,(tr,va) in enumerate(kf.split(X,y),1):
        # build terms: smooth for all + te for selected pairs
        terms = s(0,20)
        for i in range(1, X.shape[1]): terms += s(i,20)
        for (i,j) in te_pairs: terms += te(i,j)
        gam_cv = LinearGAM(terms).gridsearch(X[tr], y[tr], lam=lam_grid)
        pva = gam_cv.predict(X[va]); oof_pred[va]=pva
        r2v = r2_score(y[va], pva); rmsev = math.sqrt(mean_squared_error(y[va], pva))
        fold_stats.append({"fold":fold,"valid_r2":float(r2v),"valid_rmse":float(rmsev),
                           "lam":flatten_lam(gam_cv.lam)})
        print(f"Fold {fold}: R2={r2v:.3f}, RMSE={rmsev:.3f}")

    cv_r2 = r2_score(y, oof_pred)
    cv_rmse = math.sqrt(mean_squared_error(y, oof_pred))
    print(f"CV (OOF) R2={cv_r2:.3f}, RMSE={cv_rmse:.3f}")
    pd.DataFrame(fold_stats).to_csv("cv_fold_metrics.csv", index=False)
    pd.DataFrame({"y_true":y, "y_pred_oof":oof_pred}).to_csv("cv_oof_predictions.csv", index=False)
    print("Saved:", os.path.abspath("cv_fold_metrics.csv"))
    print("Saved:", os.path.abspath("cv_oof_predictions.csv"))
    plot_parity(y, oof_pred, "cv_parity.png", f"Actual vs Predicted (OOF, K={args.k})")

    # ---------- Permutation importance across ALL features ----------
    perm_df = permutation_importance_cv(X, y, feature_cols, seed=args.seed, k=args.k)
    print("\nPermutation importance (top 10):")
    print(perm_df.head(10).to_string(index=False))
    plot_permutation_importance(perm_df)

    terms_final = s(0, n_splines=40)
    for i in range(1, X.shape[1]): terms_final += s(i, n_splines=40)
    for (i,j) in te_pairs: terms_final += te(i,j)
    gam = LinearGAM(terms_final).gridsearch(X, y, lam=lam_grid)
    print(gam.summary())
    #analyze_with_shap(gam, data[feature_cols])
    summary = {"target": args.target, "features": feature_cols,
               "n_rows": int(len(data)), "cv_oof_r2": float(cv_r2),
               "cv_oof_rmse": float(cv_rmse), "full_lam": flatten_lam(gam.lam),
               "top_features_for_interactions": top_feats}
    with open("gam_summary.json","w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Saved:", os.path.abspath("gam_summary.json"))

    # ---------- PDP for ALL features + effect sizes ----------
    pdp_df = pdp_effect_sizes(gam, data, feature_cols, args.target)

    # ---------- Interaction ranking (H-statistic) and 3D surfaces ----------
    all_pairs = list(itertools.combinations(feature_cols, 2))
    Hdf = interaction_h_stat(gam, data, feature_cols, all_pairs, gridN=25)
    top_pairs = Hdf.head(args.n_pairs_to_plot)[["feat_a","feat_b"]].values.tolist()
    for a,b in top_pairs:
        plot_surface3d(gam, data, feature_cols, a, b)
        plot_surface3d(gam, data, feature_cols, "Ash", "FC",
               x_label="%Ash of biomass",
               y_label="%FC of biomass")
    # ---------- Optimum search within data-driven bounds ----------
    qlo, qhi = 0.02, 0.98
    bounds = {c:(float(data[c].quantile(qlo)), float(data[c].quantile(qhi))) for c in feature_cols}
    rng = np.random.default_rng(args.seed)
    # sample uniformly within bounds for each feature
    samples = [rng.uniform(*bounds[c], size=args.n_search) for c in feature_cols]
    Xc = np.column_stack(samples)
    yhat = gam.predict(Xc); j = int(np.argmax(yhat))
    best = {feature_cols[i]: float(Xc[j,i]) for i in range(len(feature_cols))}
    best["pred_yield"] = float(yhat[j])
    try:
        lb, ub = gam.prediction_intervals(Xc[j:j+1], width=0.95).ravel()
        best["pi95_low"] = float(lb); best["pi95_high"] = float(ub)
    except Exception:
        pass
    pd.DataFrame([best]).to_csv("gam_best_params.csv", index=False)
    print("Saved:", os.path.abspath("gam_best_params.csv"))

    with open("gam_model.pkl","wb") as f: pickle.dump(gam,f)
    print("Saved:", os.path.abspath("gam_model.pkl"))

    print("\nArtifacts: cv_fold_metrics.csv, cv_oof_predictions.csv, cv_parity.png, "
          "importance_permutation.csv, importance_pdp.csv, pdp_*.png, "
          "interaction_pairs.csv, surface3d_*.png, gam_summary.json, "
          "gam_best_params.csv, gam_model.pkl")

if __name__ == "__main__":
    main()
