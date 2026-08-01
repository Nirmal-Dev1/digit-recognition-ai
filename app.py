from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join("models", "digit_model.onnx")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Put digit_model.onnx inside the models/ folder.")

sess = ort.InferenceSession(MODEL_PATH)
INPUT_NAME = sess.get_inputs()[0].name


@app.route("/")
def home():
    return render_template("index.html")


def preprocess_like_mnist(img):
    """
    Matches how MNIST digits are actually formatted:
    1. crop tightly to the drawn strokes (bounding box)
    2. scale so the digit's longest side fits in a 20x20 box (aspect preserved)
    3. paste onto a blank 28x28 canvas
    4. shift so the digit's center of mass sits at the image center
    Without this, a small or off-center digit gets squished/shifted by a
    plain resize and the model reads it as the wrong shape.
    """
    arr = np.array(img).astype(np.float32)

    # bounding box of the drawn (non-black) pixels
    coords = np.argwhere(arr > 20)
    if coords.size == 0:
        # blank canvas, nothing drawn
        return np.zeros((28, 28), dtype=np.float32)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img.crop((int(x0), int(y0), int(x1), int(y1)))

    # scale longest side to 20px, preserve aspect ratio
    w, h = cropped.size
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    # paste centered onto a 28x28 canvas
    canvas = Image.new("L", (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    # shift so center of mass lands at the image center (MNIST convention)
    arr = np.array(canvas).astype(np.float32)
    total = arr.sum()
    if total > 0:
        ys, xs = np.indices(arr.shape)
        cy = (ys * arr).sum() / total
        cx = (xs * arr).sum() / total
        shift_y = int(round(14 - cy))
        shift_x = int(round(14 - cx))
        arr = np.roll(arr, shift_y, axis=0)
        arr = np.roll(arr, shift_x, axis=1)

    return arr


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image field in request"}), 400

        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read())).convert("L")

        arr = preprocess_like_mnist(img)
        arr = (arr / 255.0).astype(np.float32).reshape(1, 28, 28, 1)

        out = sess.run(None, {INPUT_NAME: arr})[0]
        pred = int(np.argmax(out))
        probs = out[0].tolist()

        return jsonify({"prediction": pred, "probs": probs})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # host="0.0.0.0" so it's reachable from a phone on the same network
    app.run(host="0.0.0.0", port=5000, debug=True)
