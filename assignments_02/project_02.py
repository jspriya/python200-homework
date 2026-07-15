import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------- Task1 - Load and Explore -------

df = pd.read_csv("student_performance_math.csv", sep=";")

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("student_performance_math.csv", sep=";")

# Explore the dataset
print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# Plot distribution of final grades
plt.figure(figsize=(8, 5))
plt.hist(df["G3"], bins=21)

plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.savefig("outputs/g3_distribution.png")
plt.show()

# ------ Task2 - Preprocess the Data ----------

# Print original shape
print("Original dataset shape:", df.shape)

# Remove students who did not take the final exam
df_filtered = df[df["G3"] > 0].copy()

# Print new shape
print("Filtered dataset shape:", df_filtered.shape)

# Print number of removed rows
print("Rows removed:", len(df) - len(df_filtered))

# We remove G3 = 0 rows because these students missed the final exam.
# A zero here does not represent academic performance, so including these
# rows would mix together two different phenomena:
#   1. Students who performed poorly.
#   2. Students who never took the exam.
# This would distort the regression model and the relationships it learns.

# Correlation before filtering

corr_original = df["absences"].corr(df["G3"])

# Correlation after filtering

corr_filtered = df_filtered["absences"].corr(df_filtered["G3"])

print("\nCorrelation between absences and G3")
print("Original dataset: ", corr_original)
print("Filtered dataset: ", corr_filtered)


# Convert yes/no columns to 1/0

yes_no_cols = [
    "schoolsup",
    "internet",
    "higher",
    "activities"
]

for col in yes_no_cols:
    df_filtered[col] = df_filtered[col].map({
        "yes": 1,
        "no": 0
    })

# Convert sex column
df_filtered["sex"] = df_filtered["sex"].map({
    "F": 0,
    "M": 1
})

print("\nFirst few rows after encoding:")
print(df_filtered.head())

# ---------- Scatter plot for original dataset --------------

plt.figure(figsize=(8, 5))
plt.scatter(df["absences"], df["G3"], alpha=0.7)

plt.title("Absences vs Final Grade (Original Dataset)")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")

plt.savefig("outputs/absences_vs_g3_original.png")
plt.show()


# -------- Scatter plot for filtered dataset (G3 > 0 only) ----------

plt.figure(figsize=(8, 5))
plt.scatter(df_filtered["absences"], df_filtered["G3"], alpha=0.7)

plt.title("Absences vs Final Grade (Filtered Dataset)")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")

plt.savefig("outputs/absences_vs_g3_filtered.png")
plt.show()

# In the original dataset, many students with G3 = 0 had extremely high
# absence counts because they never took the final exam. Their zero grades
# reflected non-participation rather than academic performance. Removing
# these students allows us to study the relationship between absences and
# achievement among students who actually completed the course.


# -------- Task3 - Exploratory Data Analysis ---------------

# -------- Compute and sort correlations ----------

# Numeric columns to examine

numeric_cols = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "absences",
    "freetime",
    "goout",
    "Walc"
]

print("Correlation with G3 (sorted):")

correlations = {}

for col in numeric_cols:
    corr = df_filtered[col].corr(df_filtered["G3"])
    correlations[col] = corr

sorted_corrs = sorted(correlations.items(), key=lambda x: x[1])

for feature, corr in sorted_corrs:
    print(f"{feature:12s}: {corr:.3f}")

# 'failures' has the strongest correlation with G3
# Students who have failed classes before tend to have lower final grades.

# Common observations:
# studytime usually has only a moderate positive correlation rather than a huge one.
# absences may be weaker than expected after filtering.
# Medu and Fedu often show a small positive relationship.
# traveltime is often nearly unrelated.

# ------- Visualization 1: Failures vs G3 ----------

plt.figure(figsize=(8,5))

plt.scatter(
    df_filtered["failures"],
    df_filtered["G3"],
    alpha=0.7
)

plt.title("Previous Failures vs Final Grade")
plt.xlabel("Number of Previous Failures")
plt.ylabel("Final Grade (G3)")

plt.savefig("outputs/failures_vs_g3.png")
plt.show()

# Students with more previous failures generally earn lower final grades.
# The decline is quite noticeable, making failures one of the strongest
# predictors in the dataset.

# ------- Visualization 2: Boxplot by study time ---------

# plt.figure(figsize=(8,5))

df_filtered.boxplot(column="G3", by="studytime",figsize=(8,5))

plt.title("Final Grade by Study Time")
plt.suptitle("")
plt.xlabel("Study Time Category")
plt.ylabel("Final Grade (G3)")

plt.savefig("outputs/g3_by_studytime.png")
plt.show()

# Median grades increase somewhat with study time, although the spread
# within each group remains large.

# --------- Task 4 - Baseline Model -------------

# Use failures as the only feature
X = df_filtered[["failures"]]
y = df_filtered["G3"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

# Print results
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("RMSE:", rmse)
print("R²:", r2)

# Interpretation:
# The slope tells us how much the predicted final grade changes for each
# additional previous failure.
#
# The slope is  -1.4275. This tells us that every additional past failure, the model predicts 
# that a student's final grade decreases by about 1.43 points on a 0–20 scale.
# So failures have a negative relationship with final grades

# The intercept is 11.93 (predicted grade when failures = 0)
# A student with no failures is predicted to have a final grade of about 11.9/20.

# The RMSE is 2.96

# RMSE (Root Mean Squared Error) tells us how far off the model's predictions typically are.
# Since grades are on a 0–20 scale:
# The model's predictions are off by about 3 grade points on average which is a fairly large error, 
# especially for a simple model.

# R²: 0.089
# R² tells us how much of the variation in grades is explained by the model. It tells us that
# Failures alone explain only about 8.9% of the differences in final grades among students.
# The remaining ~91% is explained by other factors, such as:

# study time
# absences
# previous grades (G1, G2)
# family background
# motivation
# school suppor etc.

# EDA showed that students with more failures tend to have lower G3 grades, indicating a negative relationship.
#  However, the correlation appears to be weak because failures alone explain only about 9% of the variation 
# in final grades. Therefore, the R² value of 0.089 was about what I expected from the EDA, since failures 
# are related to performance but are not the only factor affecting grades.

# ---------- Task 5 - Build the full model ------------

# Features selected from the Feature Guide
feature_cols = [
    "failures",
    "Medu",
    "Fedu",
    "studytime",
    "higher",
    "schoolsup",
    "internet",
    "sex",
    "freetime",
    "activities",
    "traveltime"
]

# Create feature matrix and target
X = df_filtered[feature_cols].values
y = df_filtered["G3"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and fit model
model_full = LinearRegression()
model_full.fit(X_train, y_train)

# Predictions
y_train_pred = model_full.predict(X_train)
y_test_pred = model_full.predict(X_test)

# Calculate metrics
train_r2 = model_full.score(X_train, y_train)
test_r2 = model_full.score(X_test, y_test)

rmse = np.sqrt(np.mean((y_test_pred - y_test) ** 2))

# Print results
print("Train R²:", train_r2)
print("Test R²:", test_r2)
print("Test RMSE:", rmse)

print("\nFeature Coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Full model interpretation:
#
# Train R²: 0.175
# Test R²: 0.154
# Test RMSE: 2.855
#
# Adding more features improved the test R² from about 0.089 to 0.154.
# This means the additional background and behavioral features help the model
# explain more variation in grades, but the improvement is modest. Student
# performance depends on many factors that are not included in this dataset.
#
# The train and test R² values are close (0.175 vs 0.154), which suggests the
# model is not heavily overfitting. The model performs similarly on unseen
# data as it does on training data.

# Production model decision:
#
# If deploying this model, I would keep failures, studytime, higher, internet,
# and parental education because they show meaningful relationships with G3
# and have reasonable practical explanations.
#
# I would consider dropping freetime and activities because their coefficients
# are close to zero and they add little predictive value. I would also review
# schoolsup carefully because its negative coefficient likely reflects
# selection bias rather than the true impact of receiving support.

# train/test gap is small:

# Train R² = 0.175
# Test R² = 0.154
# Difference = 0.021 - The model is not memorizing the training data

# ----------- Task 6 - Evaluate and Summarize --------------
 

import matplotlib.pyplot as plt

# Generate predictions from the full model on the test set
y_pred = model_full.predict(X_test)

# Create predicted vs actual scatter plot
plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    y_test,
    alpha=0.7
)

# Add diagonal reference line (perfect predictions)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Grade (G3)")
plt.ylabel("Actual Grade (G3)")

# Save plot
plt.savefig("outputs/predicted_vs_actual.png", bbox_inches="tight")

plt.show()

# Points above the diagonal line represent cases where the actual grade was
# higher than the predicted grade. The model underestimated the student's
# performance.
#
# Points below the diagonal line represent cases where the actual grade was
# lower than the predicted grade. The model overestimated the student's
# performance.
#
# The errors appear to be spread across the grade range rather than being
# concentrated only at the high or low end. However, because the model has a
# relatively low R², predictions are not tightly clustered around the diagonal.
# The model captures general patterns but still has considerable uncertainty.

# Model summary 
# After removing students with G3 = 0 (students who did not take the final
# exam), the filtered dataset contained 357 students. The test set contained
# 72 students (20% of the filtered dataset).
#
# The final model achieved:
# Test RMSE: 2.855
# Test R²: 0.154
#
# An RMSE of about 2.86 means that the model's predictions are typically wrong
# by about 3 grade points on a 0-20 grading scale. For example, if the model
# predicts a student will score 12, the actual score may commonly be around
# 9-15. This is a meaningful error because a few points can change a student's
# grade category.
#
# The R² of 0.154 means the model explains about 15% of the variation in final
# grades using the available background and behavioral features. The remaining
# variation is likely due to factors not included in the dataset, such as
# motivation, individual ability, teaching quality, or earlier academic
# knowledge.

# --------- Neglected Feature - The power of G1 -----------

# Add G1 as a feature to the full model
feature_cols_with_g1 = [
    "failures",
    "Medu",
    "Fedu",
    "studytime",
    "higher",
    "schoolsup",
    "internet",
    "sex",
    "freetime",
    "activities",
    "traveltime",
    "G1"
]

# Create feature matrix and target
X_g1 = df_filtered[feature_cols_with_g1].values
y = df_filtered["G3"].values

# Split data
X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1,
    y,
    test_size=0.2,
    random_state=42
)

# Create and fit model
model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

# Calculate test R²
test_r2_g1 = model_g1.score(X_test_g1, y_test_g1)

print("Test R² with G1 included:", test_r2_g1)

# Plain-language summary:
#
# After removing students with G3 = 0 (students who did not take the final
# exam), the filtered dataset contained 357 students. The test set contained
# 72 students.
#
# The full model achieved a test R² of 0.154 and a test RMSE of 2.855.
# This means the model explains about 15% of the variation in final grades.
# The typical prediction error is about 2.9 points on a 0-20 grading scale.
# For example, if the model predicts a student will score 12, the actual grade
# may commonly be around 9-15.
#
# The largest positive coefficient was internet (+0.834). This means that,
# holding other features constant, students with internet access at home were
# predicted to have final grades about 0.8 points higher. The second largest
# positive coefficient was higher (+0.610), suggesting that students planning
# to pursue higher education tended to have higher predicted grades.
#
# The largest negative coefficient was schoolsup (-2.062), followed by
# failures (-1.145). The negative schoolsup coefficient was surprising because
# additional academic support might be expected to improve grades. However,
# this likely reflects that students receiving extra support are already
# struggling academically. The model is capturing who receives support rather
# than showing that support causes lower grades.
#
# The train R² (0.175) and test R² (0.154) are close, suggesting that the model
# is not overfitting and performs similarly on unseen data.
#
# One surprising result was that adding many demographic and behavioral
# features only improved prediction modestly. This shows that final grades are
# influenced by many factors not captured in this dataset, such as motivation,
# prior knowledge, learning environment, and teacher effects.
#
# Adding G1 as a feature greatly improved performance, increasing the test R²
# to 0.749. However, this does not mean G1 causes G3. Both grades measure
# academic performance in the same course, so G1 is a strong indicator of
# existing student performance. This model can help identify students who may
# struggle after the first grading period, but it cannot support early
# intervention before G1 is available. Earlier intervention would require
# using information available before grades are recorded, such as attendance,
# study habits, engagement, and prior academic history.