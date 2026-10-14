from pathlib import Path
import joblib
import gradio as gr
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

def load_bundle(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            "Models not found. First run: python train_models.py"
        )
    return joblib.load(path)

cancer_bundle = load_bundle("cancer_model.pkl")
wine_bundle = load_bundle("wine_model.pkl")

def predict(bundle, values):
    row = pd.DataFrame([values], columns=bundle["features"])
    prediction = bundle["model"].predict(row)[0]
    probabilities = bundle["model"].predict_proba(row)[0]
    confidence = max(probabilities) * 100
    return f"Prediction: {prediction}\nModel confidence: {confidence:.2f}%"

with gr.Blocks(title="Logistic Regression Lab") as demo:
    gr.Markdown("# Logistic Regression — ANN Lab Task")
    gr.Markdown(
        "Two classification datasets with separate trained Logistic Regression models. "
        "Enter values in the same units as the dataset."
    )
    with gr.Tab("Breast Cancer Classification"):
        gr.Markdown("Predicts a model class of benign or malignant. This is an educational model, not a medical diagnosis.")
        cancer_inputs = [
            gr.Number(label="Mean radius", value=14.0),
            gr.Number(label="Mean texture", value=19.0),
            gr.Number(label="Mean perimeter", value=90.0),
            gr.Number(label="Mean area", value=600.0),
            gr.Number(label="Mean smoothness", value=0.10),
            gr.Number(label="Mean compactness", value=0.10),
            gr.Number(label="Mean concavity", value=0.08),
            gr.Number(label="Mean symmetry", value=0.18),
        ]
        cancer_output = gr.Textbox(label="Result", lines=3)
        gr.Button("Predict Cancer Class").click(
            fn=lambda *x: predict(cancer_bundle, x),
            inputs=cancer_inputs, outputs=cancer_output
        )
    with gr.Tab("Wine Classification"):
        gr.Markdown("Predicts one of the three wine classes in the dataset.")
        wine_inputs = [
            gr.Number(label="Alcohol", value=13.0),
            gr.Number(label="Malic acid", value=2.0),
            gr.Number(label="Ash", value=2.3),
            gr.Number(label="Alcalinity of ash", value=19.0),
            gr.Number(label="Magnesium", value=100.0),
            gr.Number(label="Flavanoids", value=2.5),
        ]
        wine_output = gr.Textbox(label="Result", lines=3)
        gr.Button("Predict Wine Class").click(
            fn=lambda *x: predict(wine_bundle, x),
            inputs=wine_inputs, outputs=wine_output
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7870)
