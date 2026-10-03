# Heart Disease Prediction with Logistic Regression

A machine learning classification project that predicts the **AHD (heart disease) outcome** from patient health features using a reproducible Scikit-learn pipeline.

## Project Overview

This notebook works with a heart-disease dataset containing **303 patient records**. The workflow prepares numeric and categorical features, trains a Logistic Regression model, and evaluates performance on unseen test data.

> **Note:** This is an educational machine learning project and is not intended for medical diagnosis.

## What I Built

- Converted the target variable `AHD` into a binary classification label
- Handled missing values with Scikit-learn imputers
- Standardized numeric features with `StandardScaler`
- One-hot encoded categorical features
- Used an **80/20 stratified train-test split**
- Built an end-to-end `Pipeline` with Logistic Regression
- Evaluated the model with accuracy, precision, recall, F1 score, ROC-AUC, and a confusion matrix

## Model Performance

### Test Set

| Metric | Score |
|---|---:|
| Accuracy | 0.885 |
| Precision | 0.839 |
| Recall | 0.929 |
| F1 Score | 0.881 |
| ROC-AUC | 0.960 |

**Confusion matrix**

```text
[[28  5]
 [ 2 26]]
```

The model correctly identified 26 positive cases in the test set while missing 2 positive cases.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter / Google Colab

## Repository Files

- `Lab_27_Heart_Disease.ipynb` — complete preprocessing, training, and evaluation workflow
- `requirements.txt` — Python dependencies

The notebook expects a file named `Heart.csv`. The dataset is not currently included in this repository.

## Run Locally

```bash
pip install -r requirements.txt
jupyter notebook
```

Then open `Lab_27_Heart_Disease.ipynb` and make sure `Heart.csv` is available in the working directory.

## Skills Demonstrated

Machine Learning · Classification · Logistic Regression · Data Preprocessing · Feature Encoding · Model Evaluation · Scikit-learn · Python
