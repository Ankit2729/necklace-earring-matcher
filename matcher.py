"""
matcher.py
----------
Core engine for the Necklace -> Earrings visual-matching prototype.

Approach (classical computer-vision, no training required):

1. FOREGROUND ISOLATION
   Product photos here sit on varied backgrounds (plain pink, dark brown,
   fabric, wood, etc). A flat colour-distance mask breaks on gradient /
   textured backgrounds, so instead we isolate the jewellery by its
   EDGE DENSITY: the piece itself is full of fine metal/gem detail (lots
   of edges), while the backdrop is smooth or softly blurred (few edges).
   We run Canny edge detection, dilate/close the edges into blobs, and
   keep the significant connected components as the foreground mask.

2. FEATURE EXTRACTION (computed only over the masked foreground pixels)
   - Colour: per-channel HSV histograms (Hue/Saturation/Value).
     This captures metal tone (gold vs silver vs rose-gold) and gemstone
     colour (emerald green, ruby red, pearl white, kundan/white stones...).
   - Texture: uniform Local Binary Pattern (LBP) histogram on the greyscale
     foreground. This captures how intricate/filigreed vs smooth-and-bold
     the piece's surface pattern is, independent of colour.

3. SIMILARITY
   Each image becomes one feature vector (colour histogram + texture
   histogram). Necklace vs. earring similarity = weighted sum of cosine
   similarities (70% colour, 30% texture by default) between the two
   vectors. All earrings are ranked and the top-K are returned.

This intentionally avoids a heavyweight pretrained deep embedding model
(e.g. CLIP/ResNet) so the prototype has zero large downloads and runs
instantly on CPU, while still directly reasoning about the visual
properties (colour + pattern) a human stylist would use to pair jewellery.
Swapping in a deep embedding later is straightforward - see README.
"""

import os
import glob
import csv
import numpy as np
import cv2
from skimage.feature import local_binary_pattern

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "images")
CSV_PATH = os.path.join(os.path.dirname(__file__), "candidate_dataset.csv")

COLOR_WEIGHT = 0.7
TEXTURE_WEIGHT = 0.3


# ---------------------------------------------------------------------
# Foreground mask
# ---------------------------------------------------------------------
def foreground_mask(img_bgr, blur=5, low=30, high=100, dilate_iter=3):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges = cv2.Canny(gray, low, high)

    kernel = np.ones((9, 9), np.uint8)
    dense = cv2.dilate(edges, kernel, iterations=dilate_iter)
    dense = cv2.morphologyEx(dense, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    h, w = gray.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dense, connectivity=8)
    keep = np.zeros_like(dense)
    min_area = 0.004 * h * w
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255

    # Fallback: if segmentation collapses (too little / too much), use whole image
    frac = keep.mean() / 255.0
    if frac < 0.02 or frac > 0.985:
        keep = np.ones((h, w), np.uint8) * 255

    return (keep > 0).astype(np.uint8)


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------
def color_feat(img_bgr, mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    parts = []
    for ch, bins, rng in zip(range(3), [24, 16, 16], [[0, 180], [0, 256], [0, 256]]):
        h = cv2.calcHist([hsv], [ch], mask, [bins], rng)
        h = cv2.normalize(h, h, norm_type=cv2.NORM_L1).flatten()
        parts.append(h)
    return np.concatenate(parts)


def texture_feat(img_bgr, mask, P=8, R=1):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    vals = lbp[mask > 0]
    if vals.size == 0:
        vals = lbp.flatten()
    hist, _ = np.histogram(vals, bins=np.arange(0, P + 3), density=True)
    return hist


def extract_features(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    mask = foreground_mask(img)
    return {
        "color": color_feat(img, mask),
        "texture": texture_feat(img, mask),
        "mask": mask,
        "image": img,
    }


def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------
# Catalog loading + index
# ---------------------------------------------------------------------
def load_catalog(csv_path=CSV_PATH, image_dir=IMAGE_DIR):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            row["path"] = os.path.join(image_dir, row["image_file"])
            rows.append(row)
    return rows


class JewelryIndex:
    """Loads the catalog and pre-computes features for every image once."""

    def __init__(self, csv_path=CSV_PATH, image_dir=IMAGE_DIR):
        self.catalog = load_catalog(csv_path, image_dir)
        self._feat_cache = {}
        for row in self.catalog:
            self._feat_cache[row["id"]] = extract_features(row["path"])

    def items_by_type(self, product_type):
        return [r for r in self.catalog if r["product_type"].lower() == product_type.lower()]

    @property
    def necklaces(self):
        return self.items_by_type("Necklace")

    @property
    def earrings(self):
        return self.items_by_type("Earrings")

    def recommend(self, necklace_id, top_k=3,
                   color_weight=COLOR_WEIGHT, texture_weight=TEXTURE_WEIGHT):
        """Return top_k earrings ranked by visual similarity to necklace_id."""
        if necklace_id not in self._feat_cache:
            raise KeyError(f"Unknown id: {necklace_id}")
        n_feat = self._feat_cache[necklace_id]

        results = []
        for row in self.earrings:
            e_feat = self._feat_cache[row["id"]]
            score = (color_weight * cosine_sim(n_feat["color"], e_feat["color"]) +
                     texture_weight * cosine_sim(n_feat["texture"], e_feat["texture"]))
            results.append((score, row))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def recommend_for_image(self, image_path, top_k=3,
                             color_weight=COLOR_WEIGHT, texture_weight=TEXTURE_WEIGHT):
        """Same as recommend(), but for an arbitrary uploaded necklace image
        (not necessarily already in the catalog)."""
        n_feat = extract_features(image_path)
        results = []
        for row in self.earrings:
            e_feat = self._feat_cache[row["id"]]
            score = (color_weight * cosine_sim(n_feat["color"], e_feat["color"]) +
                      texture_weight * cosine_sim(n_feat["texture"], e_feat["texture"]))
            results.append((score, row))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
