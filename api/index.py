import os
import tempfile

from flask import Flask, jsonify, request, send_from_directory

from matcher import JewelryIndex

app = Flask(__name__)
@app.get("/")
def home():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return send_from_directory(project_root, "index.html")
@app.get("/images/<path:filename>")
def serve_image(filename):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(project_root, "images")
    return send_from_directory(images_dir, filename)
# Load the jewellery catalog once when the server starts.
index = JewelryIndex()


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "Necklace-Earring Matcher API is running"
    })


@app.get("/api/catalog")
def catalog():
    necklaces = [
        {
            "id": row["id"],
            "product_type": row["product_type"],
            "image_file": row["image_file"],
            "image_url": f"/images/{row['image_file']}"
        }
        for row in index.necklaces
    ]

    return jsonify({
        "necklaces": necklaces
    })


@app.post("/api/recommend")
def recommend():
    try:
        top_k = int(request.form.get("top_k", 3))
        color_weight = float(request.form.get("color_weight", 0.7))

        # Keep the value in a sensible range.
        top_k = max(1, min(top_k, 15))
        color_weight = max(0.0, min(color_weight, 1.0))
        texture_weight = 1.0 - color_weight

        necklace_id = request.form.get("necklace_id")

        # Option 1: use a necklace already in the inventory.
        if necklace_id:
            results = index.recommend(
                necklace_id=necklace_id,
                top_k=top_k,
                color_weight=color_weight,
                texture_weight=texture_weight
            )

        # Option 2: use an uploaded necklace image.
        elif "image" in request.files:
            uploaded_file = request.files["image"]

            if not uploaded_file.filename:
                return jsonify({
                    "error": "No image selected."
                }), 400

            suffix = os.path.splitext(uploaded_file.filename)[1] or ".jpg"

            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:
                    uploaded_file.save(temp_file)
                    temp_path = temp_file.name

                results = index.recommend_for_image(
                    image_path=temp_path,
                    top_k=top_k,
                    color_weight=color_weight,
                    texture_weight=texture_weight
                )

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        else:
            return jsonify({
                "error": "Provide either necklace_id or an image."
            }), 400

        recommendations = []

        for score, row in results:
            recommendations.append({
                "id": row["id"],
                "product_type": row["product_type"],
                "image_file": row["image_file"],
                "image_url": f"/images/{row['image_file']}",
                "score": round(float(score), 4)
            })

        return jsonify({
            "recommendations": recommendations,
            "top_k": top_k,
            "color_weight": color_weight,
            "texture_weight": texture_weight
        })

    except KeyError as exc:
        return jsonify({
            "error": f"Unknown necklace ID: {exc.args[0]}"
        }), 400

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 500