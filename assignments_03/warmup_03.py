
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# ------------ Preprocessing ---------------
# ------------ Q1 -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape: ", y_test.shape)

# ---------- Q2 ----------------
scaler = StandardScaler()

# Fit only on training data to avoid leaking information from the test set
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Mean of each column in X_train_scaled:")
print(X_train_scaled.mean(axis=0))

# The scaler is fitted only on X_train to prevent data leakage by not using
# information from the test set during preprocessing.

# ------------ KNN ---------------
# ------------ Q1 ----------------

knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------- Q2 ---------------

knn_scaled = KNeighborsClassifier(n_neighbors=5)

# Train using the scaled data
knn_scaled.fit(X_train_scaled, y_train)

# Predict on the scaled test data
y_pred_scaled = knn_scaled.predict(X_test_scaled)

# Evaluate the model
print("Accuracy (scaled data):", accuracy_score(y_test, y_pred_scaled))

# Scaling made little or no difference for this dataset because the Iris
# features are already measured on similar scales (all in centimeters).
# KNN uses distances between points, so scaling becomes much more important
# when features have very different ranges.

# ---------- Q3 -------------

knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(knn, X_train, y_train, cv=5)

print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())

# Five-fold cross-validation provides a more trustworthy estimate of model
# performance than a single train/test split because the model is evaluated
# on five different test sets instead of just one. This reduces the chance
# that the measured accuracy is unusually high or low due to a particular
# random split of the data.

# --------------- Q4 -------------------

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)

    print(f"k = {k:2d}, Mean CV Accuracy = {scores.mean():.3f}")

# Based on the cross-validation results, I would choose k = 5 because it
# achieved the highest mean accuracy. Cross-validation evaluates the model
# on multiple train/test splits, making this a more reliable choice than
# selecting k based on a single split.
# Since there is a tie etween k=5 and k =7, I will choose k=5 because 
# it uses fewer neighbors while achieving the same performance.

# --------------- Classifier Evaluation --------------
# Q1

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()

plt.title("KNN Confusion Matrix")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.show()

# The model achieves perfect classification accuracy on this dataset, 
# so it does not confuse any pair of species.

# ---------- Decision Trees ------------
# ----------- Q1 --------------

tree = DecisionTreeClassifier(max_depth=3, random_state=42)

# Train the model
tree.fit(X_train, y_train)

# Make predictions
y_pred_tree = tree.predict(X_test)

# Evaluate the model
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree))

# The Decision Tree achieved an accuracy of 96.7%, while the KNN model
# achieved 100% accuracy on the test set. KNN performed slightly better,
# correctly classifying all test samples, whereas the Decision Tree made
# one incorrect prediction.

# Scaling would not be expected to affect the Decision Tree's performance.
# Decision Trees make decisions by splitting data based on feature thresholds
# rather than calculating distances between data points, so they generally
# produce the same results with scaled or unscaled features.

# ------------- Logistic Regression and Regularization -------------
# Q1

c_values = [0.01, 1.0, 100]

for c in c_values:
    # Use lbfgs solver (or default) to support multi-class natively without wrappers or deprecated args
    model = LogisticRegression(
        C=c,
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X_train_scaled, y_train)

    # Directly compute total coefficient magnitude
    coefficient_sum = np.abs(model.coef_).sum()

    print(f"C = {c}")
    print("Total coefficient magnitude:", coefficient_sum)
    print()
       

# As C increases, the total coefficient magnitude increases significantly.
# With C=0.01, the model applies strong regularization, forcing the
# coefficients to remain small (total magnitude ≈ 1.96). As C increases to
# 1.0 and 100, regularization becomes weaker, allowing the model to learn
# larger coefficients (12.48 and 37.89). This shows that regularization
# prevents overfitting by penalizing large coefficients and encouraging a
# simpler model.

# --------- PCA -------------

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting 

# Q1

# Print shapes
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)

# Create a 1-row subplot showing one example of each digit
fig, axes = plt.subplots(1, 10, figsize=(12, 2))

for digit in range(10):
    # Find the first image belonging to this digit
    index = np.where(y_digits == digit)[0][0]

    axes[digit].imshow(images[index], cmap="gray_r")
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")

plt.tight_layout()

plt.savefig("outputs/sample_digits.png")
plt.show()

# ------------  Q2  -------------

# Create PCA model (keeps all components)
pca = PCA()

# Fit PCA and transform the data into principal component scores
pca.fit(X_digits)

scores = pca.transform(X_digits)

# Create scatter plot using first two principal components
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    scores[:, 0],
    scores[:, 1],
    c=y_digits,
    cmap="tab10",
    s=10
)

plt.colorbar(scatter, label="Digit")

plt.title("PCA 2D Projection of Digits")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.savefig("outputs/pca_2d_projection.png")
plt.show()

# Same-digit images tend to form clusters in the PCA 2D space, although
# there is some overlap between digits. PCA captures the major patterns in
# the images, but reducing 64 dimensions to only 2 dimensions loses some
# information needed to perfectly separate all digit classes.

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()

    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]

    return reconstruction.reshape(8, 8)

# ------------  Q3  -------------
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', markersize=3)
plt.axhline(y=0.80, color='r', linestyle=':', label='80% Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.grid(True)
plt.legend()
plt.savefig("outputs/pca_variance_explained.png")
plt.show()

# To explain 80% of the variance, approximately 13 to 15 components are needed.

# -------------- Q4 ---------------------

n_values = [2, 5, 15, 40]

fig, axes = plt.subplots(
    len(n_values) + 1,   # Original row + four PCA rows
    5,
    figsize=(10, 10)
)

# First row: original images
for i in range(5):
    axes[0, i].imshow(images[i], cmap="gray_r")
    axes[0, i].set_title(f"Original {y_digits[i]}")
    axes[0, i].axis("off")

# Reconstructed images
for row, n in enumerate(n_values, start=1):

    for col in range(5):
        reconstructed = reconstruct_digit(
            col,
            scores,
            pca,
            n
        )

        axes[row, col].imshow(
            reconstructed,
            cmap="gray_r"
        )

        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_ylabel(
                f"n={n}",
                rotation=0,
                labelpad=25
            )


plt.tight_layout()

plt.savefig("outputs/pca_reconstructions.png")
plt.show()
# The digits become clearly recognizable around n=15 principal components.
# With only 2 or 5 components, the images are blurry because important pixel
# information is lost. Increasing to 40 components produces images very close
# to the originals. This matches the variance curve because most of the
# important information is captured by the first few components, and the curve
# begins to level off after the major components.
