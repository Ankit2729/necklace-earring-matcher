# Necklace → Matching Earrings Recommender

A small prototype that takes a necklace (selected from the provided
inventory, or an uploaded photo) and recommends the top visually-matching
earrings from the candidate inventory.

## What's included

```
project/
├── candidate_dataset.csv   # provided dataset (id, product_type, image_file)
├── images/                 # provided jewellery images (5 necklaces, 15 earrings)
├── matcher.py               # core matching engine (segmentation + features + similarity)
├── cli_demo.py               # CLI script — no UI needed, saves collage PNGs
├── app.py                   # Streamlit web app (interactive UI)
├── requirements.txt
└── outputs/                  # example collages produced by cli_demo.py
```

## How to run

```bash
pip install -r requirements.txt

# Option A — CLI (prints ranked matches, saves a labelled collage image per necklace)
python cli_demo.py                 # runs for all 5 necklaces
python cli_demo.py --id N01 --top 3

# Option B — interactive web app
streamlit run app.py
```

In the Streamlit app you click a necklace thumbnail (or upload your own
necklace photo) and the top-K earrings appear alongside a similarity score.
Colour/texture weighting is adjustable from the sidebar.

## Demo / Assignment Coverage

- Input: select one of the provided necklaces or upload a necklace image.
- Candidate inventory: recommendations are restricted to the 15 provided earring images.
- Output: top-K matching earrings with similarity scores.
- Adjustable matching: colour vs. texture weighting can be changed in the sidebar.
- The CLI demo can generate recommendation collages for all 5 provided necklaces.

## Approach — how images are compared

This is a **classical computer-vision pipeline**, not a trained/fine-tuned
model — chosen because it's transparent, fast on CPU, needs no downloads,
and its reasoning is easy to explain (which matters more here than
squeezing out extra accuracy).

**1. Foreground isolation.**
The product photos sit on very different backgrounds (plain pink, dark
brown, fabric, wood...). A naive colour-based background removal breaks on
gradient backgrounds, so instead each image is segmented by **edge
density**: jewellery is full of fine metal/gem detail (many edges) while
the backdrop is smooth or blurred (few edges). We run Canny edge detection,
dilate the edges into blobs, keep the significant connected components, and
use that as a foreground mask. (Falls back to using the whole image if
segmentation collapses.)

**2. Feature extraction**, computed only over the masked foreground pixels:

- **Colour** — per-channel HSV histograms. This captures metal tone (gold /
  silver / rose-gold) and gemstone colour (emerald green, ruby red, kundan
  white, pearl, etc.) — the strongest visual cue for "does this pair with
  that necklace."
- **Texture** — a uniform Local Binary Pattern (LBP) histogram on the
  greyscale foreground, capturing whether a piece is intricate/filigreed vs.
  bold and smooth, independent of colour.

**3. Similarity & ranking.**
Each image reduces to one feature vector. A necklace is compared against
every candidate earring using a weighted cosine similarity
(default 70% colour + 30% texture), and the highest-scoring earrings are
returned as recommendations.

## Technologies used

- Python 3
- OpenCV (`opencv-python-headless`) — edge detection, morphology, HSV
  histograms
- scikit-image — Local Binary Pattern texture features
- NumPy — vector math / cosine similarity
- Streamlit — interactive UI

## Limitations & how I'd extend this

- The edge-density segmentation is a heuristic — a few images with busy
  backgrounds (e.g. patterned fabric) segment less cleanly, and its
  imperfect mask carries into the colour/texture histograms. A learned
  foreground/background segmenter (e.g. `rembg`/U²-Net) would be more robust.
- With more time/compute budget, the color+texture histograms could be
  swapped for (or blended with) a **pretrained deep embedding** — e.g.
  CLIP (`open_clip`) or a torchvision ImageNet backbone (ResNet/MobileNet) —
  to also capture higher-level shape/style similarity, not just colour and
  local texture. The `matcher.py` interface (`extract_features` /
  `cosine_sim`) is written so that swap only touches one function.
- Only 5 necklaces / 15 earrings are in this inventory, so ranking quality
  is easy to eyeball but not stress-tested at scale; the same pipeline
  scales to a larger catalog by caching feature vectors (already done via
  `JewelryIndex`) and, past a few thousand items, adding an approximate
  nearest-neighbour index (e.g. FAISS).
