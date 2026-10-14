from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

CONFIG = {
    "cancer": {
        "file": "breast_cancer.csv",
        "target": "diagnosis",
        "model_file": "cancer_model.pkl",
    },
    "wine": {
        "file": "wine_classification.csv",
        "target": "wine_class",
        "model_file": "wine_model.pkl",
    },
}

def train_one(name, config):
    df = pd.read_csv(DATA_DIR / config["file"])
    X = df.drop(columns=[config["target"]])
    y = df[config["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=3000))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\\n" + "=" * 55)
    print(f"DATASET: {name.upper()}")
    print(f"Rows: {len(df)} | Features: {X.shape[1]}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    bundle = {
        "model": model,
        "features": list(X.columns),
        "target": config["target"],
        "classes": list(model.classes_),
    }
    joblib.dump(bundle, MODEL_DIR / config["model_file"])
    print(f"Saved model: models/{config['model_file']}")

if __name__ == "__main__":
    for dataset_name, settings in CONFIG.items():
        train_one(dataset_name, settings)
    print("\\nBoth Logistic Regression models are trained and saved.")
