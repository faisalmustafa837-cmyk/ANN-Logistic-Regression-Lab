from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
app = Flask(__name__)

def load_bundle(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError("Models not found. Run python train_models.py first.")
    return joblib.load(path)

bundles = {
    "cancer": load_bundle("cancer_model.pkl"),
    "wine": load_bundle("wine_model.pkl"),
}

DEFAULTS = {
    "cancer": [14.0, 19.0, 90.0, 600.0, 0.10, 0.10, 0.08, 0.18],
    "wine": [13.0, 2.0, 2.3, 19.0, 100.0, 2.5],
}

@app.route("/", methods=["GET", "POST"])
def home():
    selected = request.form.get("dataset", "cancer")
    result = None
    confidence = None
    error = None
    if selected not in bundles:
        selected = "cancer"

    if request.method == "POST":
        bundle = bundles[selected]
        try:
            values = [float(request.form[f"feature_{i}"]) for i in range(len(bundle["features"]))]
            row = pd.DataFrame([values], columns=bundle["features"])
            result = str(bundle["model"].predict(row)[0])
            confidence = f"{max(bundle['model'].predict_proba(row)[0]) * 100:.2f}%"
        except (ValueError, KeyError):
            error = "Please enter a valid number in every field."

    bundle = bundles[selected]
    fields = [
        {"name": name, "label": name.replace("_", " ").title(),
         "default": DEFAULTS[selected][i]}
        for i, name in enumerate(bundle["features"])
    ]
    return render_template(
        "index.html", selected=selected, fields=fields,
        result=result, confidence=confidence, error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
