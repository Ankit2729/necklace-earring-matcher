"""
app.py
------
Streamlit prototype UI.

Run with:
    streamlit run app.py

Lets the user pick one of the provided necklaces (or upload their own
necklace photo) and shows the top-matching earrings from the inventory,
ranked by visual similarity, with a similarity score for each.
"""
import streamlit as st
from PIL import Image
import tempfile
import os

from matcher import JewelryIndex

st.set_page_config(page_title="Necklace -> Earrings Matcher", page_icon="💍", layout="wide")


@st.cache_resource
def get_index():
    return JewelryIndex()


index = get_index()

st.title("💍 Necklace → Matching Earrings")
st.caption(
    "Select a necklace from the inventory and get the top visually-matching "
    "earrings, ranked by a colour + texture similarity score."
)

with st.sidebar:
    st.header("Options")
    top_k = st.slider("Number of earrings to recommend", 1, 6, 3)
    color_weight = st.slider("Colour weight", 0.0, 1.0, 0.7, 0.05)
    texture_weight = round(1 - color_weight, 2)
    st.write(f"Texture weight: **{texture_weight}**")
    st.divider()
    uploaded = st.file_uploader("...or upload your own necklace photo", type=["jpg", "jpeg", "png"])

st.subheader("1. Choose a necklace")

necklaces = index.necklaces
cols = st.columns(len(necklaces))
selected_id = st.session_state.get("selected_id", necklaces[0]["id"])

for col, row in zip(cols, necklaces):
    with col:
        st.image(row["path"], use_container_width=True)
        if st.button(row["id"], key=f"pick_{row['id']}", use_container_width=True):
            selected_id = row["id"]
            st.session_state["selected_id"] = selected_id

st.session_state["selected_id"] = selected_id

st.divider()
st.subheader("2. Top matching earrings")

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    left, right = st.columns([1, 3])
    with left:
        st.image(tmp_path, caption="Your uploaded necklace", use_container_width=True)
    results = index.recommend_for_image(tmp_path, top_k=top_k,
                                         color_weight=color_weight, texture_weight=texture_weight)
else:
    chosen = next(r for r in necklaces if r["id"] == selected_id)
    left, right = st.columns([1, 3])
    with left:
        st.image(chosen["path"], caption=f"Selected: {chosen['id']}", use_container_width=True)
    results = index.recommend(selected_id, top_k=top_k,
                               color_weight=color_weight, texture_weight=texture_weight)

with right:
    rcols = st.columns(len(results))
    for rcol, (score, row) in zip(rcols, results):
        with rcol:
            st.image(row["path"], use_container_width=True)
            st.metric(row["id"], f"{score:.3f}")

st.divider()
with st.expander("How does the matching work?"):
    st.markdown(
        """
        1. **Foreground isolation** – each product photo is segmented from its
           background using edge density (jewellery is detailed/high-edge,
           backgrounds are smooth), via OpenCV Canny + morphology.
        2. **Colour features** – HSV histograms over the foreground pixels
           capture metal tone (gold / silver / rose-gold) and gemstone colour.
        3. **Texture features** – a Local Binary Pattern histogram captures
           how intricate/filigreed vs. bold-and-smooth the piece is.
        4. **Similarity** – necklace and earring feature vectors are compared
           with a weighted cosine similarity (colour + texture), and the
           highest-scoring earrings are returned.
        """
    )
