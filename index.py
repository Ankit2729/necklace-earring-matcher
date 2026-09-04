import os
import sys
import tempfile

from flask import Flask, request, jsonify

# Allow importing matcher.py from the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from matcher import JewelryIndex

app = Flask(__name__)

# Build the jewelry index once per serverless instance
index = JewelryIndex()


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Necklace-Earring Matcher API is running"
    })


@app.route("/api/recommend", methods=["POST"])
def recommend():
    try:
        top_k = int(request.form.get("top_k", 3))
        color_weight = float(request.form.get("color_weight", 0.7))
        texture_weight = 1.0 - color_weight

        top_k = max(1, min(top_k, 6))
        color_weight = max(0.0, min(color_weight, 1.0))

        # ---------------------------------------------------------
        # Option 1: User uploaded a necklace image
        # ---------------------------------------------------------
        uploaded_file = request.files.get("image")

        if uploaded_file and uploaded_file.filename:
            suffix = os.path.splitext(uploaded_file.filename)[1] or ".jpg"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp:
                uploaded_file.save(tmp.name)
                tmp_path = tmp.name

            try:
                results = index.recommend_for_image(
                    tmp_path,
                    top_k=top_k,
                    color_weight=color_weight,
                    texture_weight=texture_weight
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # ---------------------------------------------------------
        # Option 2: User selected a necklace from inventory
        # ---------------------------------------------------------
        else:
            necklace_id = request.form.get("necklace_id")

            if not necklace_id:
                return jsonify({
                    "error": "Please provide either an image or necklace_id."
                }), 400

            results = index.recommend(
                necklace_id,
                top_k=top_k,
                color_weight=color_weight,
                texture_weight=texture_weight
            )

        recommendations = []

        for score, row in results:
            recommendations.append({
                "id": row["id"],
                "product_type": row["product_type"],
                "image_file": row["image_file"],
                "image_url": "/images/" + row["image_file"],
                "score": round(float(score), 4)
            })

        return jsonify({
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500