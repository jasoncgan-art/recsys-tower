"""Interactive demo: pick a user, get two-tower recommendations.

Run:  streamlit run app.py
(Requires streamlit: pip install -r requirements.txt)
"""

import streamlit as st
import torch

from recommend import load_twotower, recommend
from dataset import load_raw


@st.cache_resource
def get_model():
    return load_twotower()


st.set_page_config(page_title="Two-Tower Movie Recommender", page_icon="🎬")
st.title("🎬 Two-Tower Movie Recommender")
st.caption("Neural retrieval demo — user & item towers trained with in-batch "
           "sampled softmax on MovieLens 100k.")

model, data, art = get_model()
interactions, movies, _ = load_raw()
title_of = dict(zip(movies["item_id"], movies["title"]))

user_id = st.selectbox("User", sorted(data["user_idx"].keys()), index=0)
top_n = st.slider("How many recommendations?", 5, 25, 10)

if st.button("Recommend", type="primary"):
    liked = (interactions[interactions["user_id"] == user_id]
             .sort_values("timestamp", ascending=False)["item_id"].head(5))
    with st.expander("What this user liked recently"):
        for mid in liked:
            st.write("•", title_of.get(mid, mid))
    st.subheader("Recommended for you")
    for rank, (iidx, score) in enumerate(
            recommend(model, data, art["temperature"], user_id, top_n), 1):
        mid = int(data["item_ids"][iidx])
        st.write(f"**{rank}.** {title_of.get(mid, mid)} "
                 f"`score {score:.3f}`")
