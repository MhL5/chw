import pandas as pd
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
    y = y.map({1: 0, 2: 1})

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
n_features = X.shape[1]
n_classes = len(np.unique(y))

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
        ("select", SelectKBest(score_func=f_classif, k=min(7, n_features)))
    ]),

    "StandardScaler + PCA(5)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(5, n_features), random_state=42))
    ]),

    "StandardScaler + PCA(10)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(10, n_features), random_state=42))
    ]),

    "StandardScaler + LDA": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(n_components=min(n_classes - 1, n_features)))
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
            scoring=scoring,
            n_jobs=-1
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
output_dir = "./result"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "IVUS_model_results_with_PCA_LDA.csv")

if os.path.exists(output_file):
    os.remove(output_file)

results_df.to_csv(output_file, index=False)

print(f"\n✅ Results saved to {output_file}")





# 
# 
# -------------------------
# Visualization
# -------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder for figures
fig_dir = "./result/figures"

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# -------------------------
# Top 15 Accuracy Chart
# -------------------------
top_acc = results_df.head(15)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=top_acc,
    x="Accuracy",
    y="Classifier",
    palette="viridis"
)

plt.title("Top 15 Models by Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Model")

plt.tight_layout()

acc_path = os.path.join(fig_dir, "top15_accuracy.png")
plt.savefig(acc_path, dpi=300)
plt.close()


# -------------------------
# Top 15 F-score Chart
# -------------------------
top_f1 = results_df.sort_values(
    by="F-score",
    ascending=False
).head(15)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=top_f1,
    x="F-score",
    y="Classifier",
    palette="magma"
)

plt.title("Top 15 Models by F-score")
plt.xlabel("F-score")
plt.ylabel("Model")

plt.tight_layout()

f1_path = os.path.join(fig_dir, "top15_fscore.png")
plt.savefig(f1_path, dpi=300)
plt.close()


# -------------------------
# Heatmap of Metrics
# -------------------------
heatmap_df = results_df.copy()

heatmap_df = heatmap_df.set_index("Classifier")

plt.figure(figsize=(14, 18))

sns.heatmap(
    heatmap_df,
    annot=True,
    cmap="YlGnBu",
    fmt=".3f"
)

plt.title("Model Performance Heatmap")

plt.tight_layout()

heatmap_path = os.path.join(fig_dir, "performance_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()


# -------------------------
# PCA Visualization
# -------------------------
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_vis = PCA(n_components=2)
X_pca = pca_vis.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Label": y
})

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Label",
    palette="Set1"
)

plt.title("PCA Projection")

plt.tight_layout()

pca_path = os.path.join(fig_dir, "pca_projection.png")
plt.savefig(pca_path, dpi=300)
plt.close()


# -------------------------
# LDA Visualization
# -------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda_vis = LinearDiscriminantAnalysis(n_components=1)

X_lda = lda_vis.fit_transform(X_scaled, y)

lda_df = pd.DataFrame({
    "LD1": X_lda[:, 0],
    "Label": y
})

plt.figure(figsize=(10, 4))

sns.histplot(
    data=lda_df,
    x="LD1",
    hue="Label",
    kde=True,
    palette="Set1",
    bins=30
)

plt.title("LDA Projection")

plt.tight_layout()

lda_path = os.path.join(fig_dir, "lda_projection.png")
plt.savefig(lda_path, dpi=300)
plt.close()


print("\n✅ Charts saved in:", fig_dir)
