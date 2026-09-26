# 🍿 StreamHop — same DNA, new platform

Your watch history is the launchpad, not the cage.

StreamHop is a **cross-platform recommendation engine** with a twist on the
usual formula: instead of keeping you inside one platform watching more of the
same, it finds titles with the same **story DNA** — genres, directors, era,
pacing, mood — on **streaming platforms you don't usually watch**, and
**explains every pick** in plain English.

**Live demo:** *(link after deployment)* — pick movies you love, set your home
platforms, turn the serendipity dial, and get your next obsession with receipts.

## Why it's different

| Typical recommender | StreamHop |
|---|---|
| One platform, more of the same | 4 platforms (Netflix, Hulu, Prime Video, Disney+) |
| Black-box scores | Every pick explains itself |
| Filter bubble as a feature | Serendipity dial + bubble-burst meter |
| "People also watched" | "Same DNA as *Inception* — both heist-thrillers, now on Disney+" |

## How it works

1. **Taste DNA** — your star ratings become a weighted profile over 132 content
   features (genres ×3 weight, directors, decade, runtime, IMDb tier, age rating).
2. **Hybrid scoring** — story-DNA cosine similarity fused with item-item
   collaborative filtering trained on 100,000 real MovieLens ratings. Each signal
   is normalized within the candidate pool so the sparse CF signal isn't drowned out.
3. **Platform hop** — candidates must stream somewhere *outside* your home platforms.
4. **Serendipity dial** — Maximal Marginal Relevance re-ranking trades relevance
   for diversity, with a bonus for hopping to platforms you haven't seen yet.
5. **Receipts** — each recommendation names its anchor title, shared DNA, and gets
   flagged 🌀 **BUBBLE-BURSTER** when it ventures into genres you've never watched.

## Offline evaluation (MovieLens-100k, 120 users, leave-one-5★-out)

| Method | Hit@10 | NDCG@10 | Diversity | Platforms covered |
|---|---|---|---|---|
| Popularity | 0.367 | 0.190 | 0.697 | 3.82 |
| Content-only | 0.042 | 0.023 | 0.176 | 3.09 |
| Collaborative filtering | **0.442** | **0.243** | 0.685 | 3.67 |
| **StreamHop (hybrid + MMR)** | 0.392 | 0.227 | 0.636 | **3.91** |

The hybrid beats the popularity baseline on accuracy, keeps 89% of pure-CF
ranking quality, and covers the **widest spread of platforms** — the product
thesis, verified.

## Project structure

```
streamhop/
├── app/app.py            # Streamlit demo
├── src/
│   ├── build_catalog.py  # match MovieLens↔OTT, engineer DNA features, CF matrix
│   ├── recommender.py    # taste profiles, hybrid scoring, MMR, explanations
│   └── evaluate.py       # offline evaluation protocol
├── artifacts/            # prebuilt catalog, DNA matrix, CF similarity
├── data/                 # MovieLens-100k + streaming-platform catalog (see below)
└── requirements.txt
```

## Data

- Ratings: [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
  (100k ratings, 943 users, 1,682 movies)
- Catalog: "Movies on Netflix, Prime Video, Hulu and Disney+"
  ([Kaggle](https://www.kaggle.com/datasets/ruchi798/movies-on-netflix-prime-video-hulu-and-disney),
  16,744 titles with platform flags, genres, directors, IMDb)
- 210 MovieLens titles fuzzy-matched to the catalog for the CF signal;
  content-based DNA covers all 16,744.

Rebuild everything: `python src/build_catalog.py && python src/evaluate.py`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Limitations & next steps

- Catalog is movies-only and US-centric; TV shows would 10× the candidate pool.
- Explanations are template-based — an LLM layer could make them conversational.
- Taste DNA is static; session-based modeling could capture mood ("tonight I want…").
- This is a research/educational demo, not affiliated with any streaming service.

Built by Sree Divya — CS @ University of Houston.
