ANN LAB TASK: LOGISTIC REGRESSION WITH TWO DATASETS

Datasets included:
1. datasets/breast_cancer.csv — Wisconsin Breast Cancer dataset, 8 selected numeric features and diagnosis target (benign/malignant).
2. datasets/wine_classification.csv — Wine Recognition dataset, 6 selected numeric features and wine_class target (class_0/class_1/class_2).

Files:
- train_models.py: preprocesses, trains, evaluates and saves both Logistic Regression pipelines.
- gradio_app.py: Gradio interface with tabs for both models.
- flask_app.py: Flask web application supporting both datasets.
- templates/index.html: Flask page.

Run on Windows:
1. Install Python 3.10+ if needed.
2. Open Command Prompt in this project folder.
3. Install libraries:
   pip install pandas numpy scikit-learn gradio flask joblib
4. Train both models:
   python train_models.py
5. Start Gradio (in one terminal):
   python gradio_app.py
   Open the local URL printed by Gradio.
6. Start Flask (in a separate terminal):
   python flask_app.py
   Open http://127.0.0.1:5000

Important:
- Train models before starting either interface.
- Keep the datasets folder, models folder, Python files and templates folder together.
- Logistic Regression is a classification algorithm despite the similar name.
- CSVs are based on public scikit-learn versions of the Wisconsin Breast Cancer and Wine Recognition datasets, with selected features included for a manageable interface.
