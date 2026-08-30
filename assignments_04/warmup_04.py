import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib


os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------  ROC and AUC  ------------

# -----------  Q1  --------------------

# Logistic Regression (raw/unscaled data)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

lr_probs = lr_model.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)

print("Logistic Regression AUC:", lr_auc)


# KNN (scaled data)
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

knn_pipeline.fit(X_train, y_train)

knn_probs = knn_pipeline.predict_proba(X_test)[:, 1]
knn_auc = roc_auc_score(y_test, knn_probs)

print("KNN AUC:", knn_auc)

# KNN has the higher AUC (0.9394) compared to Logistic Regression (0.7060).
# This indicates that KNN is better at separating the positive and negative
# classes across all possible classification thresholds. Because AUC measures
# the model's ability to rank positive instances higher than negative ones,
# KNN has stronger overall discriminative performance on this dataset,
# independent of any specific classification threshold.

# ----------  Q2  ---------------

# Compute ROC curve values
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)

# Plot ROC curves
plt.figure(figsize=(8, 6))

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC = {lr_auc:.4f})"
)

plt.plot(
    knn_fpr,
    knn_tpr,
    label=f"KNN (AUC = {knn_auc:.4f})"
)

# Random classifier
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")

plt.savefig("outputs/roc_comparison.png")
plt.show()

# Find the FPR corresponding to TPR closest to 0.80

lr_idx = np.argmin(np.abs(lr_tpr - 0.80))
knn_idx = np.argmin(np.abs(knn_tpr - 0.80))

print(f"Logistic Regression: TPR = {lr_tpr[lr_idx]:.3f}, FPR = {lr_fpr[lr_idx]:.3f}")
print(f"KNN:                 TPR = {knn_tpr[knn_idx]:.3f}, FPR = {knn_fpr[knn_idx]:.3f}")

# At a TPR of approximately 0.80, KNN has a much lower FPR than Logistic
# Regression (about 0.04 vs. 0.55). This means that if the goal is to correctly
# identify around 80% of the positive cases, KNN would generate far fewer false
# positives (false alarms) than Logistic Regression. Therefore, KNN is the
# better choice when maintaining a high detection rate while minimizing false
# alarms.

# ------------- Q3  ----------------

# ROC values for Logistic Regression
lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_probs)

best_threshold = None
best_f1 = -1
best_tpr = None
best_fpr = None

for i, threshold in enumerate(lr_thresholds):
    y_pred = (lr_probs >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_tpr = lr_tpr[i]
        best_fpr = lr_fpr[i]

print(f"Best Threshold: {best_threshold:.4f}")
print(f"TPR: {best_tpr:.4f}")
print(f"FPR: {best_fpr:.4f}")
print(f"F1 Score: {best_f1:.4f}")

# The optimal threshold (0.2757) is lower than the default threshold of 0.5.
# Using a lower threshold causes the model to classify more instances as
# positive, which increases the true positive rate and improves the F1 score,
# although it also increases the false positive rate. In real applications, a
# threshold lower than 0.5 is useful when missing positive cases is more
# costly than generating false positives, such as in disease screening, fraud
# detection, or security monitoring.

# --------------  GridSearchCV -----------

# --------------- Q1 ---------------------

# Pipeline with StandardScaler and Logistic Regression
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, random_state=42))
])

# Parameter grid
param_grid = {
    "logreg__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc"
)

grid_search.fit(X_train, y_train)

# Best estimator predictions
best_probs = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
best_test_auc = roc_auc_score(y_test, best_probs)

print("Best C:", grid_search.best_params_["logreg__C"])
print("Best CV AUC:", grid_search.best_score_)
print("Test AUC of Best Estimator:", best_test_auc)

# The grid search selected C = 100.0 instead of the default value of 1.0,
# so it did not choose the value I would have guessed by default. However,
# the test AUC changed from 0.7060 to 0.7057, a decrease of about 0.0003.
# This difference is extremely small, indicating that tuning C did not
# meaningfully improve the model's performance on the test set.

# ------------------ Q2 -----------------------

# Pipeline with StandardScaler and Decision Tree
dt_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("tree", DecisionTreeClassifier(random_state=42))
])

# Parameter grid
param_grid = {
    "tree__max_depth": [2, 3, 5, 8, None]
}

# Grid Search
dt_grid = GridSearchCV(
    estimator=dt_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc"
)

dt_grid.fit(X_train, y_train)

# Test AUC of the best estimator
dt_probs = dt_grid.best_estimator_.predict_proba(X_test)[:, 1]
dt_test_auc = roc_auc_score(y_test, dt_probs)

print("Best max_depth:", dt_grid.best_params_["tree__max_depth"])
print("Best CV AUC:", dt_grid.best_score_)
print("Test AUC:", dt_test_auc)

# The Decision Tree achieved a much higher test AUC (0.9354) compared to the
# best Logistic Regression model from Q1 (0.7057). Based on AUC alone, I would
# bring the Decision Tree into further development because it provides much
# better class separation performance on this dataset.
#
# However, AUC is not the only factor to consider when selecting a model.
# I would also evaluate metrics such as precision, recall, F1 score, model
# interpretability, training and inference time, risk of overfitting, and the
# business cost of false positives versus false negatives before making a
# final decision.

# ------------------  Q3  ----------------------

# Extract CV results from Decision Tree grid search
cv_results = dt_grid.cv_results_

# Create a list of parameter values, mean AUC, and std AUC
results = []

for params, mean_score, std_score in zip(
    cv_results["params"],
    cv_results["mean_test_score"],
    cv_results["std_test_score"]
):
    results.append(
        (
            params["tree__max_depth"],
            mean_score,
            std_score
        )
    )

# Sort from best to worst mean CV AUC
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

# Print results
print("max_depth | Mean CV AUC | Std CV AUC")
print("-------------------------------------")

for depth, mean_auc, std_auc in results_sorted:
    print(f"{depth} | {mean_auc:.4f} | {std_auc:.4f}")

# The max_depth values of 5 and 3 have relatively similar mean CV AUC scores
# (0.9165 and 0.9024), but max_depth=3 has a slightly lower standard
# deviation (0.0191 vs. 0.0213), indicating more consistent performance across
# cross-validation folds. 
# If I had to choose between these two models, I would
# consider max_depth=5 because it has the higher mean AUC and only a small
# increase in variability. However, if stability across different datasets
# were more important, max_depth=3 could be preferred because it is slightly
# more consistent.

# ---------------- Joblib ---------------
# ----------------  Q1  -----------------

# Save the best Logistic Regression pipeline from GridSearch Q1
best_lr_pipe = grid_search.best_estimator_

joblib.dump(best_lr_pipe, "models/warmup_model.pkl")

# Load the saved model
loaded_clf = joblib.load("models/warmup_model.pkl")

# Compare predictions
original_preds = best_lr_pipe.predict(X_test)
loaded_preds = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"

print("Predictions match. Model saved and loaded successfully.")

# If only the Logistic Regression model were saved without the scaler, the
# loaded model would receive unscaled X_test data. Since the model was trained
# on standardized features, using unscaled inputs could produce incorrect
# predictions because the feature distributions would not match what the model
# learned. Saving the entire pipeline ensures that the same preprocessing steps
# are applied before making predictions.

# ---  Q2: Simulated prediction script ---
# Load the saved model fresh from disk
loaded_model = joblib.load("models/warmup_model.pkl")

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

# Predict classes and probabilities
new_predictions = loaded_model.predict(new_samples)
new_probabilities = loaded_model.predict_proba(new_samples)[:, 1]

# Print results
for i, (pred, prob) in enumerate(zip(new_predictions, new_probabilities)):
    print(f"Sample {i + 1}:")
    print(f"Predicted class: {pred}")
    print(f"Probability of class 1: {prob:.4f}")
    print()

# The all-zeros row predicted class 1 with a probability of 0.6531. This is
# expected because the Logistic Regression model does not make predictions
# based on whether the raw feature values are zero. The Pipeline first applies
# the StandardScaler, which transforms the features based on the training data
# mean and standard deviation. The final prediction depends on the scaled
# feature values, learned coefficients, and intercept. Since the resulting
# probability was above the default 0.5 threshold, the model classified this
# sample as class 1.