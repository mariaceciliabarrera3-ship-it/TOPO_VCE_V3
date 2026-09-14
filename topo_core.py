
"""
TOPO-VCE V3 - Núcleo Matemático Completo
Consolidado y verificado por el evaluador a partir de:
- Etapa 2 (A1-A4): geometría, ya aprobada, con la corrección del estado DROP.
- Ronda de evaluación previa (A5-A7): detección VCE, estabilidad y análisis,
  ya ejecutada y validada contra demo.csv.
Un único archivo, sin lógica duplicada.
"""

from __future__ import annotations
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, skewnorm, beta
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ============================================================
# Constantes innegociables del manual
# ============================================================
SEED = 20260912
MAX_FEATURES = 20
MAX_ROWS = 5000
MIN_ROWS = 30
DEFAULT_NULL_REPS = 39
DEFAULT_FOLDS = 3


# ============================================================
# A1 / A2 — Ingesta y curación (Etapa 2, aprobada)
# ============================================================
def load_and_curate(df: pd.DataFrame) -> pd.DataFrame:
    """Ingesta, curación, filtro de varianza e imputación por mediana."""
    df_num = df.select_dtypes(include=[np.number]).copy()
    df_num.replace([np.inf, -np.inf], np.nan, inplace=True)

    valid_cols = df_num.columns[df_num.notna().mean() >= 0.65]
    df_num = df_num[valid_cols]

    variances = df_num.var(ddof=0)
    df_num = df_num.loc[:, variances > 0]

    if df_num.shape[1] < 2:
        raise ValueError("Menos de 2 variables numéricas válidas.")

    if df_num.shape[1] > MAX_FEATURES:
        top_cols = df_num.var(ddof=0).nlargest(MAX_FEATURES).index
        df_num = df_num[top_cols]

    df_curated = df_num.fillna(df_num.median())

    if len(df_curated) < MIN_ROWS:
        raise ValueError("Menos de 30 filas válidas.")
    if len(df_curated) > MAX_ROWS:
        raise ValueError("El dataset supera el límite computacional de 5000 filas.")

    return df_curated


def scale_features(df_curated: pd.DataFrame):
    """Estandariza la matriz curada. Se usa tanto para geometría como para VCE
    (los tres canales de detección son invariantes a esta transformación lineal
    por columna: Spearman por rangos, MI por cuantiles, y BIC-gain porque el
    término jacobiano de la reescala se cancela en la diferencia BIC(1)-BIC(2))."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_curated.to_numpy(float))
    return X_scaled, scaler


# ============================================================
# A3 / A4 — Proyección y geometría (Etapa 2, aprobada, DROP corregido)
# ============================================================
def project_pca(X_scaled: np.ndarray) -> np.ndarray:
    n_components = min(2, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    return pca.fit_transform(X_scaled)


def compute_geometry(X_scaled: np.ndarray, pc_coords: np.ndarray, k: int = 8) -> dict:
    n_samples = X_scaled.shape[0]
    k_effective = min(k, n_samples - 1)

    nn = NearestNeighbors(n_neighbors=k_effective + 1)
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)

    d_k = distances[:, -1]
    d_k = np.where(d_k == 0, 1e-10, d_k)

    raw_density = 1.0 / d_k
    min_d, max_d = raw_density.min(), raw_density.max()
    density = (raw_density - min_d) / (max_d - min_d) if max_d > min_d else np.zeros_like(raw_density)

    slope = np.zeros(n_samples)
    for i in range(n_samples):
        neighbor_idx = indices[i, 1:]
        d_ij = distances[i, 1:]
        d_ij = np.where(d_ij == 0, 1e-10, d_ij)
        slopes_i = np.abs(density[i] - density[neighbor_idx]) / d_ij
        slope[i] = slopes_i.max()

    curvature = np.zeros(n_samples)
    for i in range(n_samples):
        neighbor_idx = indices[i, 1:]
        curvature[i] = np.abs(density[i] - density[neighbor_idx].mean())

    q25, q75, q85_slope = np.percentile(density, 25), np.percentile(density, 75), np.percentile(slope, 85)
    states = []
    for d, s in zip(density, slope):
        # CORRECCIÓN: DROP reemplaza al estado, no se concatena.
        if s >= q85_slope:
            st = "DROP"
        elif d >= q75:
            st = "PEAK"
        elif d <= q25:
            st = "VALLEY"
        else:
            st = "PLATEAU"
        states.append(st)

    return {
        "density": density,
        "slope": slope,
        "curvature": curvature,
        "state": np.array(states, dtype=object),
        "neighbors": indices[:, 1:],
        "PC1": pc_coords[:, 0],
        "PC2": pc_coords[:, 1] if pc_coords.shape[1] > 1 else np.zeros(n_samples),
    }


# ============================================================
# A5 — Detección VCE: Canal A (Spearman), Canal B (marginal), Canal C (MI)
# Validados en la ronda de evaluación previa.
# ============================================================
def _max_spearman(y: np.ndarray, X: np.ndarray, j: int) -> float:
    vals = []
    for k in range(X.shape[1]):
        if k == j:
            continue
        r = spearmanr(y, X[:, k]).statistic
        vals.append(abs(float(r)) if np.isfinite(r) else 0.0)
    return max(vals) if vals else 0.0


def _quantile_codes(x: np.ndarray, bins: int = 8):
    x = np.asarray(x, float)
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    cuts = np.unique(np.quantile(x, qs))
    if cuts.size == 0:
        return np.zeros(len(x), dtype=np.int64), 1
    codes = np.digitize(x, cuts, right=False).astype(np.int64)
    return codes, int(cuts.size + 1)


def _plugin_mi_from_codes(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> float:
    n = len(a)
    joint = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(float)
    pxy = joint / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / (px @ py)[mask])))


def _prepare_mi_bins(X: np.ndarray, bins: int = 8):
    codes, sizes = [], []
    for j in range(X.shape[1]):
        c, s = _quantile_codes(X[:, j], bins=bins)
        codes.append(c)
        sizes.append(s)
    return codes, sizes


def _max_plugin_mi(j: int, target_codes: np.ndarray, codes: List[np.ndarray], sizes: List[int]) -> float:
    vals = []
    for k in range(len(codes)):
        if k == j:
            continue
        vals.append(_plugin_mi_from_codes(target_codes, codes[k], sizes[j], sizes[k]))
    return max(vals) if vals else 0.0


def _bic_gain(y: np.ndarray, seed: int = SEED) -> float:
    a = np.asarray(y, dtype=float).reshape(-1, 1)
    g1 = GaussianMixture(n_components=1, random_state=seed, n_init=3).fit(a)
    g2 = GaussianMixture(n_components=2, random_state=seed, n_init=3).fit(a)
    return float(g1.bic(a) - g2.bic(a))


def _fit_beta_or_skewnorm(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    n = len(y)
    lo, hi = float(np.min(y)), float(np.max(y))
    span = hi - lo

    if span <= 0:
        return "skewnorm", skewnorm.fit(y)

    eps = 1.0 / (n + 1.0)
    u = np.clip((y - lo) / span, eps, 1.0 - eps)

    try:
        ba, bb, _, _ = beta.fit(u, floc=0, fscale=1)
        loglik_b = float(np.sum(beta.logpdf(u, ba, bb)))
        aic_b = 2 * 2 - 2 * loglik_b
    except Exception:
        aic_b, ba, bb = np.inf, np.nan, np.nan

    try:
        sp = skewnorm.fit(y)
        loglik_s = float(np.sum(skewnorm.logpdf(y, *sp)))
        aic_s = 2 * 3 - 2 * loglik_s
    except Exception:
        aic_s, sp = np.inf, None

    if aic_b <= aic_s:
        return "beta", (float(ba), float(bb), lo, hi)
    return "skewnorm", tuple(float(v) for v in sp)


def _sample_marginal_null(fit, n: int, rng: np.random.Generator):
    family, params = fit
    if family == "beta":
        a, b, lo, hi = params
        u = beta.rvs(a, b, size=n, random_state=rng)
        return lo + u * (hi - lo)
    return skewnorm.rvs(*params, size=n, random_state=rng)


def empirical_p(observed: float, null_values: np.ndarray) -> float:
    return float((1 + np.sum(null_values >= observed)) / (len(null_values) + 1))


def channel_scores(X: np.ndarray, feature_names: List[str],
                    null_reps: int = DEFAULT_NULL_REPS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    observed, dep_nulls, mi_nulls, bic_nulls = [], [], [], []
    bic_reps = min(null_reps, 19)
    mi_codes, mi_sizes = _prepare_mi_bins(X, bins=8)

    for j, name in enumerate(feature_names):
        y = X[:, j]
        dep_obs = _max_spearman(y, X, j)
        mi_obs = _max_plugin_mi(j, mi_codes[j], mi_codes, mi_sizes)
        bic_obs = _bic_gain(y, seed + j)

        dep_n, mi_n, bic_n = [], [], []
        fit = _fit_beta_or_skewnorm(y)

        for b in range(null_reps):
            permutation = rng.permutation(len(y))
            yp = y[permutation]
            dep_n.append(_max_spearman(yp, X, j))
            mi_n.append(_max_plugin_mi(j, mi_codes[j][permutation], mi_codes, mi_sizes))
            if b < bic_reps:
                bic_n.append(_bic_gain(
                    _sample_marginal_null(fit, len(y), rng),
                    seed + j * 1000 + 100 + b
                ))

        observed.append({"variable": name, "dep": dep_obs, "mi": mi_obs,
                          "bic_gain": bic_obs, "null_family": fit[0]})
        dep_nulls.append(np.asarray(dep_n))
        mi_nulls.append(np.asarray(mi_n))
        bic_nulls.append(np.asarray(bic_n))

    dep_max = np.max(np.vstack(dep_nulls), axis=0)
    mi_max = np.max(np.vstack(mi_nulls), axis=0)
    bic_max = np.max(np.vstack(bic_nulls), axis=0)

    rows = []
    for i, obs in enumerate(observed):
        rows.append({
            **obs,
            "dep_null95": float(np.quantile(dep_nulls[i], 0.95)),
            "dep_p": empirical_p(obs["dep"], dep_max),
            "mi_null95": float(np.quantile(mi_nulls[i], 0.95)),
            "mi_p": empirical_p(obs["mi"], mi_max),
            "bic_null95": float(np.quantile(bic_nulls[i], 0.95)),
            "bic_p": empirical_p(obs["bic_gain"], bic_max),
        })
    return pd.DataFrame(rows)


def feature_vce_table(X: np.ndarray, feature_names: List[str],
                       null_reps: int = DEFAULT_NULL_REPS, seed: int = SEED) -> pd.DataFrame:
    scores = channel_scores(X, feature_names, null_reps=null_reps, seed=seed)
    scores["dep_q"] = scores["dep_p"]
    scores["mi_q"] = scores["mi_p"]
    scores["bic_q"] = scores["bic_p"]
    scores["dep_signal"] = (scores.dep > scores.dep_null95) & (scores.dep_p <= 0.05)
    scores["mi_signal"] = (scores.mi > scores.mi_null95) & (scores.mi_p <= 0.05)
    scores["bic_signal"] = (scores.bic_gain > scores.bic_null95) & (scores.bic_p <= 0.05)
    scores["VCE"] = scores[["dep_signal", "mi_signal", "bic_signal"]].any(axis=1)
    return scores


# ============================================================
# A6 — Estabilidad K-Fold (reutiliza el escalado global, tal como se validó)
# ============================================================
def stability(X: np.ndarray, feature_names: List[str], folds: int = DEFAULT_FOLDS,
              null_reps: int = 19, seed: int = SEED) -> dict:
    n = len(X)
    if n < MIN_ROWS:
        return {c: np.nan for c in feature_names}

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    counts = {name: 0 for name in feature_names}
    total = 0

    for fold, (_, test_idx) in enumerate(kf.split(X)):
        part = X[test_idx]
        if len(part) < MIN_ROWS:
            continue
        sc = channel_scores(part, feature_names, null_reps=null_reps, seed=seed + 100 + fold)
        total += 1
        for _, r in sc.iterrows():
            stable = bool((r.dep > r.dep_null95) or (r.mi > r.mi_null95) or (r.bic_gain > r.bic_null95))
            if stable:
                counts[r.variable] += 1

    denom = max(total, 1)
    return {k: v / denom for k, v in counts.items()}


# ============================================================
# A7 — Pipeline completo
# ============================================================
def analyze(df: pd.DataFrame, null_reps: int = DEFAULT_NULL_REPS,
            folds: int = DEFAULT_FOLDS, seed: int = SEED) -> dict:
    curated = load_and_curate(df)
    names = list(curated.columns)

    X_scaled, scaler = scale_features(curated)
    xy = project_pca(X_scaled)
    geo = compute_geometry(X_scaled, xy, k=8)

    scores = feature_vce_table(X_scaled, names, null_reps=null_reps, seed=seed)
    stab = stability(X_scaled, names, folds=folds, null_reps=19, seed=seed)
    scores["stability"] = scores.variable.map(stab)
    scores["VCE"] = scores.VCE & (scores.stability >= 0.50)

    return {
        "numeric": curated,
        "X": X_scaled,
        "xy": xy,
        "geometry": geo,
        "scores": scores,
        "stability": stab,
    }


def representative_evidence(result: dict, variable: str) -> dict:
    names = list(result["numeric"].columns)
    X = result["X"]
    geo = result["geometry"]
    scores = result["scores"].set_index("variable")

    j = names.index(variable)
    z = (X[:, j] - X[:, j].mean()) / max(X[:, j].std(), 1e-12)
    row = int(np.argmax(np.abs(z)))

    return {
        "variable": variable,
        "row": row,
        "pc1": float(result["xy"][row, 0]),
        "pc2": float(result["xy"][row, 1]),
        "density": float(geo["density"][row]),
        "slope": float(geo["slope"][row]),
        "curvature": float(geo["curvature"][row]),
        "state": str(geo["state"][row]),
        "score": scores.loc[variable].to_dict(),
    }
