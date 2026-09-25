"""Offline evaluation of StreamHop on MovieLens-100k.

Protocol: for each test user, hide one 5-star movie (must be in the matched
catalog), build the taste profile from their other 4-5 star movies, recommend
10 titles, and measure ranking quality + diversity. Compares:
  - popularity baseline
  - content-only (DNA cosine)
  - CF-only (item-item)
  - StreamHop hybrid + MMR
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from recommender import Recommender

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def dcg_at_k(rank_of_target, k):
    if rank_of_target is None or rank_of_target >= k:
        return 0.0
    return 1.0 / np.log2(rank_of_target + 2)


def main(n_users=120, top_n=10, seed=42):
    rec = Recommender()
    ratings = pd.read_csv(os.path.join(DATA, "ml-100k", "u.data"), sep="\t",
                          header=None, names=["user", "item", "rating", "ts"])
    cf = np.load(os.path.join(os.path.dirname(__file__), "..",
                              "artifacts", "cf.npz"))
    ml_ids = cf["ml_ids"]
    ml_to_full = {int(m): int(f) for m, f in
                  zip(ml_ids, cf["matched_full_idx"])}
    rng = np.random.RandomState(seed)

    # popularity baseline: rating counts per matched movie
    pop = (ratings[ratings["item"].isin(ml_to_full)]
           .groupby("item").size().sort_values(ascending=False))
    pop_full = [ml_to_full[i] for i in pop.index]

    users = ratings["user"].unique()
    rng.shuffle(users)
    tested = 0
    agg = {m: {"hr": [], "ndcg": [], "div": [], "plat": []}
           for m in ["popularity", "content", "cf", "streamhop"]}

    for u in users:
        if tested >= n_users:
            break
        ur = ratings[ratings["user"] == u]
        loved = ur[(ur["rating"] >= 4) & (ur["item"].isin(ml_to_full))]
        if len(loved) < 6:
            continue
        target_ml = loved[loved["rating"] == 5]
        if len(target_ml) == 0:
            continue
        target_full = ml_to_full[int(target_ml.sample(1, random_state=rng)["item"].iloc[0])]
        liked_full = [ml_to_full[int(i)] for i in loved["item"]
                      if ml_to_full[int(i)] != target_full][:8]
        if len(liked_full) < 3:
            continue

        profile = rec.taste_profile(liked_full)
        content = rec.X @ profile
        cf_s = rec._cf_scores(liked_full)
        has_cf = np.zeros(len(rec.cat)); has_cf[rec.matched_idx] = 1
        hybrid = 0.65 * content + 0.35 * (cf_s * has_cf + content * (1 - has_cf))

        cands = {"popularity": pop_full, "content": None, "cf": None,
                 "streamhop": None}
        order_c = np.argsort(-content)
        sub = rec.item_sim[:, [rec._match_pos[i] for i in liked_full
                               if i in rec._match_pos]].mean(1)
        cf_rank = np.argsort(-sub)
        cands["cf"] = [int(rec.matched_idx[i]) for i in cf_rank]
        cands["content"] = [int(i) for i in order_c]

        for method in agg:
            if method == "streamhop":
                recs = rec.recommend(liked_full, home_platforms=[],
                                    serendipity=0.5, top_n=top_n)
                ranked = [r["idx"] for r in recs]
            else:
                ranked = [i for i in cands[method]
                          if i not in liked_full][:top_n]
            rank = ranked.index(target_full) if target_full in ranked else None
            agg[method]["hr"].append(1.0 if rank is not None else 0.0)
            agg[method]["ndcg"].append(dcg_at_k(rank, top_n))
            idx = ranked[:top_n]
            if len(idx) > 1:
                S = rec.X[idx] @ rec.X[idx].T
                iu = np.triu_indices(len(idx), 1)
                agg[method]["div"].append(float(1 - S[iu].mean()))
            plats = {p for i in idx for p in rec.cat.loc[i, "platforms"]}
            agg[method]["plat"].append(len(plats))

        tested += 1
        if tested % 30 == 0:
            print(f"evaluated {tested} users...", flush=True)

    print(f"\n=== Results over {tested} users (P@{top_n} ~ hit rate) ===")
    print(f"{'method':<12}{'hit@10':>8}{'ndcg@10':>9}{'diversity':>11}{'platforms':>11}")
    for m, v in agg.items():
        print(f"{m:<12}{np.mean(v['hr']):>8.3f}{np.mean(v['ndcg']):>9.3f}"
              f"{np.mean(v['div']):>11.3f}{np.mean(v['plat']):>11.2f}")


if __name__ == "__main__":
    main()
