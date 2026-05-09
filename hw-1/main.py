import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
import os
from xgboost import XGBClassifier

# -------------------------
# Load data
# -------------------------
file_path = "./data/IVUS.xlsx"

df = pd.read_excel(file_path, header=None)

df.columns = [f"F{i}" for i in range(df.shape[1]-1)] + ["Label"]


# -------------------------
# Split features / target
# -------------------------

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

if set(pd.unique(y)) == {1, 2}:
    y = y.map({1:0, 2:1})

print("Unique target values:", y.unique())

# -------------------------
# Metrics
# -------------------------
def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)  # sensitivity/recall for positive class
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, sens, spec, f1

results = []

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc, sens, spec, f1 = get_metrics(y_test, y_pred)
    results.append({
        "Classifier": name,
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "F-score": f1
    })

# -------------------------
# Models
# -------------------------
base_models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    ),
    "SVM (RBF)": SVC(
        probability=True,
        kernel="rbf",
        C=2.0,
        gamma="scale",
        random_state=42,
        class_weight="balanced"  
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=7)
}

# Hybrid models
voting_model = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
        ("et", ExtraTreesClassifier(n_estimators=400, random_state=42, class_weight="balanced")),
        ("xgb", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        ))
    ],
    voting="soft"
)

stacking_model = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
        ("et", ExtraTreesClassifier(n_estimators=400, random_state=42, class_weight="balanced")),
        ("svm", SVC(probability=True, kernel="rbf", C=2.0, gamma="scale", random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
    passthrough=True
)

# -------------------------
# Train/test split first
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# Pipelines / preprocessing setups
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
# Run experiments
# -------------------------
for prep_name, prep in preprocessors.items():
    X_train_p = prep.fit_transform(X_train, y_train)
    X_test_p = prep.transform(X_test)

    for model_name, model in base_models.items():
        evaluate_model(f"{model_name} ({prep_name})", model, X_train_p, X_test_p, y_train, y_test)

    evaluate_model(f"Voting Ensemble ({prep_name})", voting_model, X_train_p, X_test_p, y_train, y_test)
    evaluate_model(f"Stacking Ensemble ({prep_name})", stacking_model, X_train_p, X_test_p, y_train, y_test)

# -------------------------
# Show results
# -------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Accuracy", "F-score"], ascending=False)

# 1. Print to console
print(results_df.to_string(index=False))

# 2. Export to CSV
output_filename = "IVUS_model_results.csv"

# Remove existing file (optional safeguard)
if os.path.exists(output_filename):
    os.remove(output_filename)

results_df.to_csv(output_filename, index=False)
print(f"\n✅ Results exported and overwritten: {output_filename}")
