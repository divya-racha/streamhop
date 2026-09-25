"""StreamHop demo: cross-platform recommendations that burst your bubble.

Tell it what you love and where you usually watch — it finds titles with the
same DNA (genres, style, era, mood) on OTHER streaming platforms, explains
every pick, and lets you dial up the serendipity.
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from recommender import PLATFORM_COLORS, PLATFORMS, Recommender  # noqa: E402

st.set_page_config(page_title="StreamHop", page_icon="\U0001f37f",
                   layout="wide")

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #3a1c71 0%, #d76d77 50%, #ffaf7b 100%);
        border-radius: 16px; padding: 2.2rem 2rem; color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0 0 0.4rem 0; font-size: 2rem; }
    .hero p { margin: 0; opacity: 0.9; font-size: 1.05rem; }
    .card {
        background: #ffffff; border: 1px solid #e6e8eb; border-radius: 14px;
        padding: 1.2rem 1.4rem; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .rec-card {
        background: #ffffff; border: 1px solid #e6e8eb; border-radius: 14px;
        padding: 1.1rem 1.3rem; margin-bottom: 0.9rem;
        border-left: 5px solid #d76d77;
    }
    .rec-title { font-size: 1.15rem; font-weight: 700; margin: 0; }
    .rec-meta { color: #666; font-size: 0.9rem; margin: 0.15rem 0 0.5rem 0; }
    .plat {
        display: inline-block; color: white; font-size: 0.75rem; font-weight: 700;
        border-radius: 20px; padding: 0.15rem 0.7rem; margin-right: 0.3rem;
    }
    .expl { font-size: 0.95rem; color: #333; margin-top: 0.5rem; }
    .burst {
        display: inline-block; background: #f3e8ff; color: #7c3aed;
        font-size: 0.78rem; font-weight: 700; border-radius: 20px;
        padding: 0.15rem 0.7rem; margin-top: 0.4rem;
    }
    .matchbar { height: 8px; border-radius: 4px; background: #eee; margin-top: 0.4rem; }
    .matchfill { height: 8px; border-radius: 4px;
        background: linear-gradient(90deg, #d76d77, #ffaf7b); }
    .footer { text-align: center; color: #999; font-size: 0.8rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>\U0001f37f StreamHop</h1>
        <p>Your watch history is the launchpad, not the cage. Tell me what you
        love — I'll find the same <b>story DNA</b> on <b>other platforms</b>,
        outside your usual genres.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine():
    return Recommender()


engine = get_engine()
cat = engine.cat

# ------------------------------------------------------------ step 1: taste ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("① What have you been loving lately?")
st.write("Search and add at least 3 movies you love. These become your taste DNA.")

if "history" not in st.session_state:
    st.session_state.history = {}  # idx -> stars

q = st.text_input("Search movies", placeholder="e.g. Inception, The Dark Knight…")
if q:
    hits = cat[cat["title"].str.contains(q, case=False, na=False)].head(8)
    for i, row in hits.iterrows():
        c1, c2 = st.columns([4, 1])
        with c1:
            st.write(f"**{row['title']}** ({int(row['year']) if row['year'] == row['year'] else '?'})"
                     f" — {', '.join(g for g in row['genres'][:3] if g != 'nan')}")
        with c2:
            if st.button("＋ Add", key=f"add_{i}"):
                st.session_state.history[i] = 5
                st.rerun()

if st.session_state.history:
    st.write("**Your watch history:**")
    for i in list(st.session_state.history):
        row = cat.loc[i]
        c1, c2, c3 = st.columns([4, 2, 1])
        with c1:
            st.write(f"🎬 {row['title']} ({int(row['year']) if row['year'] == row['year'] else '?'})")
        with c2:
            stars = st.slider("Rating", 1, 5, st.session_state.history[i],
                              key=f"stars_{i}", label_visibility="collapsed")
            st.session_state.history[i] = stars
        with c3:
            if st.button("✕", key=f"rm_{i}"):
                del st.session_state.history[i]
                st.rerun()
else:
    st.info("👆 Search above to add movies you love.")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------ step 2: platforms ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("② Where do you usually watch?")
home = st.multiselect(
    "Your home platforms (recommendations will come from everywhere else)",
    PLATFORMS, default=["Netflix"],
)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------- step 3: serendipity ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("③ How adventurous are you feeling?")
serendipity = st.slider(
    "Serendipity dial", 0.0, 1.0, 0.55, 0.05,
    help="Left = more of what you love. Right = burst the bubble: new genres, new platforms.",
    format="%.2f",
)
c1, c2 = st.columns([1, 1])
with c1:
    st.caption("🎯 More of the same")
with c2:
    st.caption("🌀 Surprise me")
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------- recommend ---
ready = len(st.session_state.history) >= 3
if not ready:
    st.warning("Add at least 3 movies to your watch history to get recommendations.")
    st.stop()

if st.button("🚀 Find my next obsession", type="primary", use_container_width=True):
    liked_idx = list(st.session_state.history)
    weights = [st.session_state.history[i] / 5 for i in liked_idx]
    with st.spinner("Scanning 16,744 titles across 4 platforms…"):
        recs = engine.recommend(liked_idx, home_platforms=home,
                                serendipity=serendipity, top_n=10,
                                weights=weights)
    st.session_state.recs = recs
    st.session_state.metrics = engine.list_metrics(recs)

if "recs" in st.session_state and st.session_state.recs:
    recs = st.session_state.recs
    st.subheader(f"Your cross-platform picks ({len(recs)})")

    m = st.session_state.metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Bubble diversity", m["diversity"])
    mc2.metric("Genres covered", m["genre_coverage"])
    mc3.metric("Bubble-bursts", f"{m['bubble_bursts']} 🌀")
    mc4.metric("Platforms", ", ".join(m["platform_spread"].keys()))

    for r in recs:
        badges = "".join(
            f'<span class="plat" style="background:{PLATFORM_COLORS[p]}">{p}</span>'
            for p in r["platforms"]
        )
        meta = f"{r['year'] or ''} · IMDb {r['imdb']:.1f} · " \
               f"{', '.join(r['genres'][:3])}" if r["imdb"] == r["imdb"] else \
               f"{r['year'] or ''} · {', '.join(r['genres'][:3])}"
        burst = ('<div><span class="burst">🌀 BUBBLE-BURSTER — new genre for you'
                 "</span></div>" if r["bubble_burster"] else "")
        st.markdown(
            f'<div class="rec-card">'
            f'<p class="rec-title">#{r["rank"]} {r["title"]}</p>'
            f'<p class="rec-meta">{meta}</p>'
            f"<div>{badges}</div>"
            f'<div class="matchbar"><div class="matchfill" '
            f'style="width:{r["match_pct"]}%"></div></div>'
            f'<p class="rec-meta">DNA match: {r["match_pct"]}%</p>'
            f'<p class="expl">{r["explanation"]}</p>'
            f"{burst}"
            f"</div>",
            unsafe_allow_html=True,
        )
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("How it works")
    st.markdown(
        "1. **Taste DNA** — your ratings become a weighted profile over 132 "
        "content features (genres, directors, era, pacing, mood).\n"
        "2. **Hybrid scoring** — story-DNA similarity fused with collaborative "
        "filtering from 100k real MovieLens ratings.\n"
        "3. **Platform hop** — candidates must stream somewhere you *don't* "
        "usually watch.\n"
        "4. **Serendipity dial** — maximal-marginal-relevance re-ranking trades "
        "pure relevance for diversity and new-platform discovery.\n"
        "5. **Receipts** — every pick explains itself: shared DNA, the anchor "
        "title it echoes, and bubble-bursts flagged honestly."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="footer">StreamHop &middot; MovieLens + streaming catalog '
    "&middot; Built by Sree Divya</div>",
    unsafe_allow_html=True,
)
