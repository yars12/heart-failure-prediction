"""Heart Failure Mortality Prediction

Educational machine-learning project based on my final project.
Expected dataset: heart_failure.csv
Target: DEATH_EVENT (0 = survived follow-up, 1 = died during follow-up)

This project is not intended for medical diagnosis or clinical use.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_PATH = "heart_failure.csv"
TARGET = "DEATH_EVENT"

sns.set_theme(style="whitegrid")


def evaluate(model_name, model, X, y):
    """Return a consistent metric row for a fitted binary classifier."""
    pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, score)
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)
        auc = roc_auc_score(y, score)
    else:
        auc = np.nan

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def main():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in {DATA_PATH}")

    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())

    # Basic EDA
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(11, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_features = X.columns.tolist()

    scaled_prep = ColumnTransformer(
        [("num", StandardScaler(), numeric_features)],
        remainder="drop",
    )

    models = {
        "LogisticRegression (baseline)": Pipeline(
            [
                ("prep", scaled_prep),
                ("model", LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
            ]
        ),
        "GaussianNB (baseline)": Pipeline(
            [("prep", scaled_prep), ("model", GaussianNB())]
        ),
        "KNN (baseline)": Pipeline(
            [("prep", scaled_prep), ("model", KNeighborsClassifier())]
        ),
        "SVC RBF (baseline)": Pipeline(
            [
                ("prep", scaled_prep),
                ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]
        ),
        "DecisionTree (baseline)": Pipeline(
            [
                ("prep", "passthrough"),
                ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForest (baseline)": Pipeline(
            [
                ("prep", "passthrough"),
                ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "GradientBoosting (baseline)": Pipeline(
            [
                ("prep", "passthrough"),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
    }

    baseline_rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        baseline_rows.append(evaluate(name, model, X_test, y_test))

    baseline = pd.DataFrame(baseline_rows).sort_values("ROC_AUC", ascending=False)
    print("\nBaseline leaderboard:\n")
    print(baseline.to_string(index=False))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rf_search = RandomizedSearchCV(
        estimator=models["RandomForest (baseline)"],
        param_distributions={
            "model__n_estimators": [200, 400, 800],
            "model__max_depth": [None, 3, 5, 8, 12],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", None],
        },
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)

    svc_search = RandomizedSearchCV(
        estimator=models["SVC RBF (baseline)"],
        param_distributions={
            "model__C": np.logspace(-2, 2, 20),
            "model__gamma": np.logspace(-4, 0, 20),
            "model__kernel": ["rbf"],
        },
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    svc_search.fit(X_train, y_train)

    tuned_rows = [
        evaluate("RandomForest (tuned)", rf_search.best_estimator_, X_test, y_test),
        evaluate("SVC RBF (tuned)", svc_search.best_estimator_, X_test, y_test),
    ]

    leaderboard = pd.concat(
        [baseline, pd.DataFrame(tuned_rows)],
        ignore_index=True,
    ).sort_values("ROC_AUC", ascending=False)

    print("\nFinal leaderboard:\n")
    print(leaderboard.to_string(index=False))

    best_model = rf_search.best_estimator_
    best_pred = best_model.predict(X_test)

    print("\nTuned Random Forest classification report:\n")
    print(classification_report(y_test, best_pred, digits=3))

    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Tuned Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    rf = best_model.named_steps["model"]
    importance = pd.DataFrame(
        {
            "feature": numeric_features,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nRandom Forest feature importance:\n")
    print(importance.to_string(index=False))

    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )

    perm_df = pd.DataFrame(
        {
            "feature": numeric_features,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    print("\nPermutation importance:\n")
    print(perm_df.to_string(index=False))


if __name__ == "__main__":
    main()
