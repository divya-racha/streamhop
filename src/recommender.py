"""StreamHop recommendation engine.

Pipeline:
  1. Taste DNA  - weighted average of DNA vectors of titles the user loves.
  2. Hybrid score - content (DNA cosine) + collaborative (item-item CF).
  3. Cross-platform filter - candidates must be watchable OFF the user's
     home platforms (that's the whole point: same DNA, new platform).
  4. MMR re-rank - serendipity dial trades relevance for diversity.
  5. Explanations - every pick gets a human-readable "why".
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
ART = os.path.join(HERE, "..", "artifacts")

PLATFORMS = ["Netflix", "Hulu", "Prime Video", "Disney+"]
PLATFORM_COLORS = {
    "Netflix": "#E50914",
    "Hulu": "#1CE783",
    "Prime Video": "#00A8E1",
    "Disney+": "#113CCF",
}


class Recommender:
    def __init__(self):
        cat = pd.read_parquet(os.path.join(ART, "catalog.parquet"))
        cat["platforms"] = cat["platforms"].str.split("|")
        cat["genres"] = cat["genres"].str.split("|")
        cat["directors"] = cat["directors"].str.split("|")
        self.cat = cat.reset_index(drop=True)

        z = np.load(os.path.join(ART, "dna_matrix.npz"))
        self.X = z["X"]
        self.feat_names = z["feat_names"].tolist()

        cf = np.load(os.path.join(ART, "cf.npz"))
        self.item_sim = cf["item_sim"]
        self.matched_idx = cf["matched_full_idx"]  # positions in catalog
        self._match_pos = {int(v): i for i, v in enumerate(self.matched_idx)}

        # genre weights of each title for explanations
        gi = [i for i, f in enumerate(self.feat_names) if f.startswith("g_")]
        self._genre_idx = gi
        self._genre_names = [self.feat_names[i][2:] for i in gi]

    # ------------------------------------------------------------- scoring ---
    def taste_profile(self, liked_idx, weights=None):
        w = np.ones(len(liked_idx)) if weights is None else np.asarray(weights)
        P = (self.X[liked_idx] * w[:, None]).sum(0)
        return P / (np.linalg.norm(P) + 1e-9)

    def _cf_scores(self, liked_idx):
        """Mean item-item similarity of every catalog title to liked titles."""
        scores = np.zeros(len(self.cat))
        lp = [self._match_pos[i] for i in liked_idx if i in self._match_pos]
        if not lp:
            return scores
        sub = self.item_sim[:, lp].mean(1)
        for full_i, s in zip(self.matched_idx, sub):
            scores[full_i] = max(s, 0)
        return scores

    def recommend(self, liked_idx, home_platforms, serendipity=0.5,
                  top_n=10, weights=None, exclude_idx=None):
        """Return list of dicts: rank, catalog idx, scores, explanation parts."""
        liked_idx = list(liked_idx)
        profile = self.taste_profile(liked_idx, weights)
        content = self.X @ profile
        cf = self._cf_scores(liked_idx)

        home = set(home_platforms)
        cand_mask = np.array([
            any(p not in home for p in plats) and i not in set(liked_idx)
            for i, plats in enumerate(self.cat["platforms"])
        ])
        if exclude_idx:
            for i in exclude_idx:
                cand_mask[i] = False
        cand = np.where(cand_mask)[0]
        if len(cand) == 0:
            return []

        # hybrid: min-max normalize each signal within the candidate pool
        # so the sparse CF signal isn't drowned by dense content scores
        cc = content[cand]
        ff = cf[cand]
        cc_n = (cc - cc.min()) / (cc.max() - cc.min() + 1e-9)
        ff_n = (ff - ff.min()) / (ff.max() - ff.min() + 1e-9)
        hybrid = np.zeros(len(self.cat), dtype=np.float32)
        hybrid[cand] = 0.55 * cc_n + 0.45 * ff_n

        # MMR re-rank: relevance vs diversity among selected,
        # with a light bonus for hopping to new platforms
        lam = 1.0 - 0.75 * serendipity
        order = cand[np.argsort(-hybrid[cand])]
        selected, remaining = [], list(order[:400])
        sel_platforms = set()
        while remaining and len(selected) < top_n:
            if not selected:
                i0 = remaining.pop(0)
                selected.append(i0)
                sel_platforms.update(self.cat.loc[i0, "platforms"])
                continue
            sim_to_sel = self.X[remaining] @ self.X[selected].T
            plat_rep = np.array([
                len(set(self.cat.loc[i, "platforms"]) & sel_platforms) /
                max(1, len(self.cat.loc[i, "platforms"]))
                for i in remaining
            ])
            mmr = (lam * hybrid[remaining]
                   - (1 - lam) * sim_to_sel.max(1)
                   - 0.12 * serendipity * plat_rep)
            best = int(np.argmax(mmr))
            chosen = remaining.pop(best)
            selected.append(chosen)
            sel_platforms.update(self.cat.loc[chosen, "platforms"])

        user_genres = self._top_genres(profile)
        liked_titles = {i: self.cat.loc[i, "title"] for i in liked_idx}
        results = []
        for rank, i in enumerate(selected, 1):
            row = self.cat.loc[i]
            new_plats = [p for p in row["platforms"] if p not in home]
            expl = self._explain(i, liked_idx, liked_titles, user_genres,
                                 new_plats)
            results.append({
                "rank": rank, "idx": i, "title": row["title"],
                "year": int(row["year"]) if pd.notna(row["year"]) else None,
                "genres": [g for g in row["genres"] if g != "nan"],
                "imdb": row["imdb"],
                "platforms": new_plats,
                "match_pct": round(float(50 + 50 * hybrid[i]), 1),
                "explanation": expl["text"],
                "bubble_burster": expl["bubble_burster"],
                "dna_overlap": expl["overlap"],
            })
        return results

    # -------------------------------------------------------- explanations ---
    def _top_genres(self, profile, k=6):
        w = profile[self._genre_idx]
        order = np.argsort(-w)[:k]
        return [self._genre_names[i] for i in order if w[i] > 0]

    def _explain(self, i, liked_idx, liked_titles, user_genres, new_plats):
        row = self.cat.loc[i]
        sims = self.X[i] @ self.X[liked_idx].T
        anchor = liked_idx[int(np.argmax(sims))]
        anchor_title = liked_titles[anchor]
        a_genres = set(g for g in self.cat.loc[anchor, "genres"] if g != "nan")
        r_genres = set(g for g in row["genres"] if g != "nan")
        shared = [g for g in r_genres if g in a_genres][:2]
        new_genres = [g for g in r_genres if g not in set(user_genres)][:2]

        parts = []
        if shared:
            parts.append(f"same DNA as *{anchor_title}* — both "
                         f"{'/'.join(shared)}")
        else:
            parts.append(f"echoes *{anchor_title}* in style and structure")
        a_dir = self.cat.loc[anchor, "directors"]
        r_dir = row["directors"]
        common_dir = [d for d in r_dir if d in a_dir and d != "nan"]
        if common_dir:
            parts.append(f"directed by {common_dir[0]}")
        text = "; ".join(parts) + "."
        if new_plats:
            text += f" Streaming on **{', '.join(new_plats)}** — outside your usual platforms."
        bubble = bool(new_genres)
        if bubble:
            text += f" 🌀 Bubble-burster: you've never watched {', '.join(new_genres)}, but the story bones match your taste."
        overlap = round(float(self.X[i] @ self.X[liked_idx].mean(0)
                              / (np.linalg.norm(self.X[liked_idx].mean(0)) + 1e-9)) * 100, 1)
        return {"text": text, "bubble_burster": bubble, "overlap": overlap}

    # ------------------------------------------------------------- metrics ---
    def list_metrics(self, recs):
        idx = [r["idx"] for r in recs]
        if len(idx) < 2:
            return {}
        S = self.X[idx] @ self.X[idx].T
        iu = np.triu_indices(len(idx), 1)
        ild = 1 - S[iu].mean()  # intra-list diversity
        genres = {g for r in recs for g in r["genres"]}
        plats = {}
        for r in recs:
            for p in r["platforms"]:
                plats[p] = plats.get(p, 0) + 1
        return {
            "diversity": round(float(ild), 3),
            "genre_coverage": len(genres),
            "platform_spread": plats,
            "bubble_bursts": sum(1 for r in recs if r["bubble_burster"]),
        }
