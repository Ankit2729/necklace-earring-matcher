"""
cli_demo.py
-----------
Command-line demo: pick a necklace, print & save the top matching earrings.

Usage:
    python cli_demo.py                 # runs for every necklace in the catalog
    python cli_demo.py --id N01        # runs for a single necklace id
    python cli_demo.py --id N01 --top 5
"""
import argparse
import os
import cv2
import numpy as np

from matcher import JewelryIndex

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def make_collage(necklace_row, results, out_path):
    tile = 220
    imgs = [cv2.imread(necklace_row["path"])]
    labels = [f"INPUT: {necklace_row['id']}"]
    for score, row in results:
        imgs.append(cv2.imread(row["path"]))
        labels.append(f"{row['id']}  {score:.3f}")

    tiles = []
    for im, label in zip(imgs, labels):
        im = cv2.resize(im, (tile, tile))
        cv2.rectangle(im, (0, 0), (tile, 28), (255, 255, 255), -1)
        cv2.putText(im, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(im)

    # divider between input and recommendations
    divider = np.full((tile, 6, 3), 40, dtype=np.uint8)
    row_img = tiles[0]
    row_img = np.hstack([row_img, divider] + tiles[1:])
    cv2.imwrite(out_path, row_img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="Necklace id from candidate_dataset.csv (e.g. N01). "
                                  "If omitted, runs for all necklaces.")
    ap.add_argument("--top", type=int, default=3, help="Number of earrings to recommend")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    index = JewelryIndex()

    targets = [n for n in index.necklaces if n["id"] == args.id] if args.id else index.necklaces
    if not targets:
        print(f"No necklace found with id={args.id}")
        return

    for necklace_row in targets:
        results = index.recommend(necklace_row["id"], top_k=args.top)
        print(f"\n{necklace_row['id']} ({necklace_row['image_file']}) -> top {args.top} earrings:")
        for score, row in results:
            print(f"   {row['id']:5s} {row['image_file']:15s} similarity={score:.3f}")

        out_path = os.path.join(OUT_DIR, f"match_{necklace_row['id']}.png")
        make_collage(necklace_row, results, out_path)
        print(f"   saved collage -> {out_path}")


if __name__ == "__main__":
    main()
