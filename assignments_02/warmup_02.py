 # --- scikit-learn API --- 
import os 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ------------- Q1 --------------

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# Create the model
model = LinearRegression()

# Fit the model to the data
model.fit(years, salary)

# Make predictions
salary_4_years = model.predict([[4]])[0]
salary_8_years = model.predict([[8]])[0]

# Print results
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salary for 4 years of experience:", salary_4_years)
print("Predicted salary for 8 years of experience:", salary_8_years)

# --------- Q2 ------------

# Original 1D array
x = np.array([10, 20, 30, 40, 50])

# Print the original shape
print("Original shape:", x.shape)

# Convert the 1D array into a 2D array with 5 rows and 1 column
x_2d = x.reshape(-1, 1)

# Print the new shape
print("New shape:", x_2d.shape)

# Scikit-learn requires X to be 2D because it expects data in the form:
# (number of samples, number of features), even when there is only one feature.

# ----------- Q3 --------------
# Generate synthetic data with 3 natural clusters

X_clusters, _ = make_blobs(
    n_samples=120, 
    centers=3, 
    cluster_std=0.8, 
    random_state=7
)

# 1. Create the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# 2. Fit the model to the data
kmeans.fit(X_clusters)

# 3. Predict the cluster label for each point
labels = kmeans.predict(X_clusters)

# Print cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Print number of points in each cluster
print("\nPoints in each cluster:")
print(np.bincount(labels))

# Create scatter plot of clusters
plt.scatter(
    X_clusters[:, 0], 
    X_clusters[:, 1], 
    c=labels,
    alpha=0.7,
    label="Data points"
)

# Plot cluster centers as black X markers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    color="black",
    s=200,
    label="Cluster centers"
)

# Add title and axis labels
plt.title("K-Means Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Save the figure
plt.savefig("outputs/kmeans_clusters.png", bbox_inches="tight")

# Display plot
plt.show()


# ------------- Linear Regression --------------------

# Synthetic data
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# ------ Q1 ---------

# Create outputs folder if it doesn't already exist
os.makedirs("outputs", exist_ok=True)

# Create scatter plot
plt.scatter(
    age,
    cost,
    c=smoker,
    cmap="coolwarm"
)

# Add title and labels
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Annual Medical Cost ($)")

# Save figure
plt.savefig("outputs/cost_vs_age.png")

# Display figure
plt.show()

# -------- Q2 -------------

# Reshape age into a 2D array for scikit-learn
X = age.reshape(-1, 1)

# Split data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    cost,
    test_size=0.2,
    random_state=42
)

# Print shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# -------- Q3 ----------

# Create and fit the Linear Regression model
model = LinearRegression()

# Train the model using training data
model.fit(X_train, y_train)

# Print slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict costs on the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

# Calculate R² score
r2 = model.score(X_test, y_test)

# Print evaluation metrics
print("RMSE:", rmse)
print("R² on the test set:", r2)

# The slope represents the estimated change in medical cost for each additional
# year of age. For example, a slope of about $200 means that medical costs
# increase by approximately $200 for every additional year of age.

# --------- Q4 ------------

# Create a feature matrix with both age and smoker status
X_full = np.column_stack([age, smoker])

# Split the data into training and test sets
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    cost,
    test_size=0.2,
    random_state=42
)

# Create and fit the new model
model_full = LinearRegression()

model_full.fit(X_train_full, y_train_full)

# Calculate and print test R²
r2_full = model_full.score(X_test_full, y_test_full)

print("Test R² with age and smoker:", r2_full)

# Compare with Question 3 R²
print("Test R² with age only:", r2)

# Print coefficients
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# The smoker coefficient represents the estimated increase in annual medical
# cost for smokers compared with non-smokers, while keeping age constant.

# -------- Q5 --------

# Make predictions using the two-feature model
y_pred_full = model_full.predict(X_test_full)

# Create predicted vs actual scatter plot
plt.scatter(
    y_pred_full,
    y_test_full,
    alpha=0.7
)

# Add diagonal reference line (perfect predictions)
plt.plot(
    [y_test_full.min(), y_test_full.max()],
    [y_test_full.min(), y_test_full.max()],
    linestyle="--"
)

# Add title and axis labels
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Medical Cost ($)")
plt.ylabel("Actual Medical Cost ($)")

# Save the figure
plt.savefig("outputs/predicted_vs_actual.png", bbox_inches="tight")

# Display plot
plt.show()

# Points above the diagonal line mean the actual cost was higher than the
# model predicted (the model underestimated the cost).
# Points below the diagonal line mean the actual cost was lower than the
# model predicted (the model overestimated the cost).