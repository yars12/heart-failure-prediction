# Heart Failure Mortality Prediction

An end-to-end machine learning classification project that predicts the **DEATH_EVENT** outcome from clinical heart-failure records.

This project was developed from my machine learning final project and demonstrates exploratory analysis, model comparison, hyperparameter tuning, evaluation, and feature interpretation.

> Educational project only. It is not intended for medical diagnosis or clinical decision-making.

## Project Highlights

- Worked with **299 patient records and 12 clinical features**
- Compared 7 classification approaches
- Used stratified train/test splitting for a consistent class balance
- Tuned Random Forest and SVM with `RandomizedSearchCV`
- Evaluated models using Accuracy, Precision, Recall, F1, ROC-AUC, and confusion matrices
- Used Random Forest feature importance and permutation importance for interpretation

## Models Compared

- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Random Forest
- Gradient Boosting

## Best Model

The tuned Random Forest produced the strongest ROC-AUC in the completed notebook run.

| Metric | Score |
|---|---:|
| Accuracy | 0.833 |
| Precision | 0.846 |
| Recall | 0.579 |
| F1 Score | 0.688 |
| ROC-AUC | 0.906 |

The baseline Random Forest was also competitive, with ROC-AUC of approximately **0.892**.

## Technologies

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

## Repository Files

- `heart_failure_mortality_prediction.py` — cleaned project workflow
- `requirements.txt` — Python dependencies

The script expects a dataset named `heart_failure.csv`.

## Run

```bash
pip install -r requirements.txt
python heart_failure_mortality_prediction.py
```

## Skills Demonstrated

Machine Learning · Classification · Model Comparison · Hyperparameter Tuning · ROC-AUC · Feature Importance · Data Visualization · Python
