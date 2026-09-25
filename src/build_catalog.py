"""Build the StreamHop catalog.

1. Engineer 'show DNA' content features for ALL OTT titles (16.7k).
2. Match MovieLens-100k titles -> OTT (for collaborative-filtering signal).
3. Compute item-item CF similarity on the matched subset.
Saves artifacts for the app.
"""
import os
import re
import difflib

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(OUT, exist_ok=True)

PLATFORMS = ["Netflix", "Hulu", "Prime Video", "Disney+"]


def norm_title(t: str) -> str:
    t = t.lower()
    # "Usual Suspects, The" -> "the usual suspects"
    t = re.sub(r",\s*(the|a|an)$", r" \1", t)
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return re.sub(r"[^a-z0-9]", "", t)


def split_title_year(ml_title: str):
    m = re.search(r"\((\d{4})\)\s*$", ml_title)
    year = int(m.group(1)) if m else None
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", ml_title).strip()
    # drop parenthetical aliases: "Seven (Se7en)" -> "Seven"
    title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    return title, year


def build_dna(df: pd.DataFrame) -> np.ndarray:
    all_genres = sorted({g for gs in df["genres"] for g in gs if g and g != "nan"})
    top_dirs = pd.Series([d for ds in df["directors"] for d in ds
                          if d and d != "nan"]).value_counts().head(80).index.tolist()

    def decade(y):
        return f"{int(y)//10*10}s" if pd.notna(y) else "unknown"

    def rt_bucket(r):
        if pd.isna(r):
            return "rt_unknown"
        return "rt_short" if r < 95 else "rt_medium" if r <= 125 else "rt_long"

    def imdb_bucket(v):
        if pd.isna(v):
            return "imdb_unknown"
        return "imdb_low" if v < 6.5 else "imdb_mid" if v < 7.5 else "imdb_high"

    def age_bucket(a):
        a = str(a)
        if "7" in a or a.strip().lower() in ("g", "all"):
            return "age_kids"
        if "18" in a:
            return "age_adult"
        return "age_teen"

    feats = []
    for _, r in df.iterrows():
        f = {}
        for g in r["genres"]:
            if g and g != "nan":
                f[f"g_{g}"] = 3.0
        for d in r["directors"][:2]:
            f[f"d_{d if d in top_dirs else 'other'}"] = 1.0
        f[f"dec_{decade(r['year'])}"] = 1.0
        f[rt_bucket(r["runtime"])] = 1.0
        f[imdb_bucket(r["imdb"])] = 1.0
        f[age_bucket(r["age"])] = 1.0
        feats.append(f)

    feat_names = sorted({k for f in feats for k in f})
    X = np.zeros((len(df), len(feat_names)), dtype=np.float32)
    fi = {n: i for i, n in enumerate(feat_names)}
    for i, f in enumerate(feats):
        for k, v in f.items():
            X[i, fi[k]] = v
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X, feat_names


def main():
    ott = pd.read_csv(os.path.join(DATA, "ott_movies.csv"))
    ott["norm"] = ott["Title"].astype(str).map(norm_title)

    full = pd.DataFrame({
        "ott_idx": np.arange(len(ott)),
        "title": ott["Title"],
        "year": pd.to_numeric(ott["Year"], errors="coerce"),
        "genres": [[g.strip() for g in str(x).split(",")] for x in ott["Genres"]],
        "directors": [[d.strip() for d in str(x).split(",")] for x in ott["Directors"]],
        "imdb": pd.to_numeric(ott["IMDb"], errors="coerce"),
        "runtime": pd.to_numeric(ott["Runtime"], errors="coerce"),
        "age": ott["Age"].astype(str),
        "platforms": [[p for p in PLATFORMS if r[p] == 1]
                      for _, r in ott.iterrows()],
    })
    full = full[full["platforms"].map(len) > 0].reset_index(drop=True)
    print(f"full OTT catalog: {len(full)} titles with platform info")

    X_full, feat_names = build_dna(full)
    print(f"DNA matrix: {X_full.shape}")

    # ---- match MovieLens ----
    by_key = {}
    for i, r in ott.iterrows():
        y = int(r["Year"]) if pd.notna(r["Year"]) else None
        by_key.setdefault((r["norm"], y), []).append(i)
    norm_list = ott["norm"].tolist()

    ml = pd.read_csv(os.path.join(DATA, "ml-100k", "u.item"), sep="|",
                     header=None, encoding="latin-1")[[0, 1]]
    ml.columns = ["ml_id", "ml_title"]

    ml_to_ott = {}
    for _, r in ml.iterrows():
        title, year = split_title_year(r["ml_title"])
        nt = norm_title(title)
        hit = by_key.get((nt, year))
        if hit:
            ml_to_ott[r["ml_id"]] = hit[0]
            continue
        cands = difflib.get_close_matches(nt, norm_list, n=3, cutoff=0.82)
        for c in cands:
            j = norm_list.index(c)
            oy = ott.loc[j, "Year"]
            if pd.notna(oy) and year and abs(int(oy) - year) <= 1:
                ml_to_ott[r["ml_id"]] = j
                break
    print(f"matched {len(ml_to_ott)}/{len(ml)} MovieLens titles")

    # map matched ott rows -> full catalog indices (keep aligned pairs only)
    ott_pos = {oi: fi for fi, oi in enumerate(full["ott_idx"])}
    valid = [(m, j) for m, j in ml_to_ott.items() if j in ott_pos]
    ml_ids = [m for m, _ in valid]
    matched_full_idx = [ott_pos[j] for _, j in valid]
    print(f"matched titles present in catalog: {len(matched_full_idx)}")

    # ---- item-item CF similarity on matched subset ----
    ratings = pd.read_csv(os.path.join(DATA, "ml-100k", "u.data"), sep="\t",
                          header=None, names=["user", "item", "rating", "ts"])
    ratings = ratings[ratings["item"].isin(ml_ids)]
    users = {u: i for i, u in enumerate(ratings["user"].unique())}
    item_pos = {m: i for i, m in enumerate(ml_ids)}
    R = np.zeros((len(users), len(ml_ids)), dtype=np.float32)
    for _, r in ratings.iterrows():
        R[users[r["user"]], item_pos[r["item"]]] = r["rating"]
    mask = R > 0
    means = np.where(mask.any(1), R.sum(1) / mask.sum(1), 0)
    Rc = np.where(mask, R - means[:, None], 0)
    nrm = np.linalg.norm(Rc, axis=0, keepdims=True) + 1e-9
    item_sim = (Rc / nrm).T @ (Rc / nrm)
    np.fill_diagonal(item_sim, 0)

    # ---- save ----
    save = full.copy()
    save["platforms"] = save["platforms"].map("|".join)
    save["genres"] = save["genres"].map(lambda gs: "|".join(g for g in gs if g != "nan"))
    save["directors"] = save["directors"].map(lambda ds: "|".join(d for d in ds if d != "nan"))
    save.to_parquet(os.path.join(OUT, "catalog.parquet"), index=False)
    np.savez_compressed(os.path.join(OUT, "dna_matrix.npz"), X=X_full,
                        feat_names=np.array(feat_names))
    np.savez(os.path.join(OUT, "cf.npz"),
             item_sim=item_sim.astype(np.float32),
             matched_full_idx=np.array(matched_full_idx, dtype=np.int64),
             ml_ids=np.array(ml_ids, dtype=np.int64))
    print("saved catalog.parquet, dna_matrix.npz, cf.npz")


if __name__ == "__main__":
    main()
