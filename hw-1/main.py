import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, confusion_matrix

from xgboost import XGBClassifier


# -------------------------
# Load dataset
# -------------------------
file_path = "./data/IVUS.xlsx"

df = pd.read_excel(file_path, header=None)

df.columns = [f"F{i}" for i in range(df.shape[1]-1)] + ["Label"]

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convert labels if needed
if set(pd.unique(y)) == {1, 2}:
    y = y.map({1:0, 2:1})

print("Target classes:", y.unique())


# -------------------------
# Custom metric: Specificity
# -------------------------
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


# Scoring metrics
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "sensitivity": make_scorer(recall_score),
    "specificity": make_scorer(specificity_score),
    "f1": make_scorer(f1_score)
}


# -------------------------
# Preprocessing pipelines
# -------------------------
preprocessors = {
    "StandardScaler": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]),
    
    "MinMaxScaler": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ]),
    
    "StandardScaler + SelectKBest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=min(7, X.shape[1])))
    ])
}


# -------------------------
# Base classifiers
# -------------------------
def get_models():

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    et = ExtraTreesClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced"
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    svm = SVC(
        probability=True,
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )

    models = {
        "Random Forest": rf,
        "Extra Trees": et,
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": xgb,
        "SVM (RBF)": svm,
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7)
    }

    models["Voting Ensemble"] = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("xgb", xgb)
        ],
        voting="soft"
    )

    models["Stacking Ensemble"] = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("svm", svm)
        ],
        final_estimator=LogisticRegression(
            max_iter=5000,
            class_weight="balanced"
        ),
        passthrough=True
    )

    return models


# -------------------------
# Cross-validation
# -------------------------
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# -------------------------
# Run experiments
# -------------------------
results = []

for prep_name, prep in preprocessors.items():

    models = get_models()

    for model_name, model in models.items():

        pipeline = Pipeline([
            ("prep", prep),
            ("model", model)
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring
        )

        results.append({
            "Classifier": f"{model_name} ({prep_name})",
            "Accuracy": np.mean(scores["test_accuracy"]),
            "Sensitivity": np.mean(scores["test_sensitivity"]),
            "Specificity": np.mean(scores["test_specificity"]),
            "F-score": np.mean(scores["test_f1"])
        })


# -------------------------
# Results
# -------------------------
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["Accuracy", "F-score"],
    ascending=False
)

print(results_df.to_string(index=False))


# -------------------------
# Export results
# -------------------------
output_file = "IVUS_model_results.csv"

if os.path.exists(output_file):
    os.remove(output_file)

results_df.to_csv(output_file, index=False)

print(f"\n✅ Results saved to {output_file}")
