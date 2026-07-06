import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy import stats

# *********************************************
#                   Pandas                    
# *********************************************

# --------------------- Pandas Q1 ----------------------
#Create the DataFrame and print the first three rows, the shape, and the data types of each column.


data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

# print(f"Number of Rows: {len(df)}") - Remove this.use for reference
# Print the first three rows
print("First three rows:")
print(df.head(3))

# Print the shape
print("\nShape of the dataframe:")
print(df.shape)

# Print the data types of each column
print("\nData types:")
print(df.dtypes)

# ------------- Pandas Q2 - filter only students who passed and have a grade above 80 -------------

filtered_df = df[(df["passed"] == True) & (df["grade"] > 80)]
print("\nStudents with grade > 80:")
print(filtered_df)

# ---------------- Pandas Q3 - Add new column - 'grade_curved' ----------------------

df['grade_curved'] = df['grade'] + 5 
print("\nUpdated dataframe:")
print(df)

# ----------------- Pandas Q4 - create new column with upper case names ------------------

df["name_upper"] = df["name"].str.upper()
print(df[['name','name_upper']])

# ----------------- Pandas Q5 - Groupby 'city' and compute mean grades ---------------------

mean_grades = df.groupby("city", as_index=False)["grade"].mean()
print("\nMean grades for each city:")
print(mean_grades)

# ------------------ Pandas Q6 - Replace 'Austin' to 'Houston' in City column -----------------

df["city"].replace("Austin", "Houston")
# Print the name and city columns
print(df[["name", "city"]])

# ------------------ Pandas Q7 - Sort by grade in descending order --------------------

sorted_df = df.sort_values(by="grade", ascending=False)

print("\nSorted by 'grade':")
# Print the top 3 rows
print(sorted_df.head(3))

# *********************************************
#               NumPy Review 
# *********************************************

# ------------------- NumPy Q1 - Create a 1D array ----------------------------

arr = np.array([10, 20, 30, 40, 50])
# Print the array
print("Array:", arr)

# Print its properties
print("Shape:", arr.shape)
print("Data type:", arr.dtype)
print("Number of dimensions:", arr.ndim)


# ------------------- NumPy Q2 - Create a 2D array -------------------------------

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print("Array:\n", arr)

# Print the shape
print("Shape:", arr.shape)

# Print the size
print("Size:", arr.size)


# ------------------ NumPy Q3 - Slice the top-left 2x2 block ----------------------

top_left = arr[:2, :2]

# Print the result
print(top_left)

# ----------------- NumPy Q4 -------------------------

# Create a 3x4 array of zeros
zeros_array = np.zeros((3, 4))

# Create a 2x5 array of ones
ones_array = np.ones((2, 5))

print("3x4 array of zeros:")
print(zeros_array)

print("\n2x5 array of ones:")
print(ones_array)

# ------------------ Numpy Q5 -  array using np.arange ----------------------

# Create the array
arr = np.arange(0, 50, 5)

# Print the array
print("Array:", arr)

# Print its properties
print("Shape:", arr.shape)
print("Mean:", arr.mean())
print("Sum:", arr.sum())
print("Standard Deviation:", arr.std())


# ----------------- NumPy Q6 - Generate 200 random values from a normal distribution -------------

arr = np.random.normal(loc=0, scale=1, size=200)

# Print the mean and standard deviation
print("Mean:", arr.mean())
print("Standard Deviation:", arr.std())

# *********************************************
#              Matplotlib Review
# *********************************************


# ----------------- Matplotlib Q1 -------------------

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

# Create the line plot
plt.plot(x, y)

# Add title and axis labels
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")

# Display the plot
plt.show()


# ------------------ Matplotlib Q2 -----------------------

subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

# Create the bar plot
plt.bar(subjects, scores)

plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")

plt.show()

# Matplotlib Q3

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

# Create scatter plots
plt.scatter(x1, y1, color="green", label="Dataset 1")
plt.scatter(x2, y2, color="purple", label="Dataset 2")

# Add title and axis labels
plt.title("Scatter Plot of Two Datasets")
plt.xlabel("x")
plt.ylabel("y")

# Add legend
plt.legend()

# Display the plot
plt.show()


# ----------------------- Matplotlib Q4 ----------------------

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

# Data for Q2
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

# Create a figure with 1 row and 2 columns of subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Left subplot: Line plot
ax[0].plot(x, y)
ax[0].set_title("Squares")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")

# Right subplot: Bar plot
ax[1].bar(subjects, scores)
ax[1].set_title("Subject Scores")
ax[1].set_xlabel("Subjects")
ax[1].set_ylabel("Scores")

# Adjust spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()

# *********************************************
#       Descriptive Statistics Review
# **********************************************

# Q1 - compute  mean, median, variance, and standard deviation

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

# Convert the list to a NumPy array
arr = np.array(data)

# Compute and print statistics
print("Mean:", np.mean(arr))
print("Median:", np.median(arr))
print("Variance:", np.var(arr))
print("Standard Deviation:", np.std(arr))

# ------------------- Q2 - Generate 500 random values from a normal distribution ----------------

scores =  np.random.normal(65, 10, 500)

# Create the histogram
plt.hist(scores, bins=20)

# Add title and axis labels
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")

plt.show()

# --------------------- Q3 - Create boxplot --------------------------

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]


plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])

# Add title and axis labels
plt.title("Score Comparison")
plt.xlabel("Groups")
plt.ylabel("Scores")

# Display the plot
plt.show()

# Q4 - Compare normal and exponential distribution

# Generate the datasets
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

# Create the boxplots
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])

plt.title("Distribution Comparison")
plt.xlabel("Distribution")
plt.ylabel("Values")

plt.show()

# The exponential distribution is more skewed to the right.
# For the normal distribution, the mean is an appropriate measure
# of central tendency because the data is approximately symmetric.

# For the exponential distribution, the median is more appropriate
# because it is less affected by the long right tail and extreme values.

# Q5 - Mean vs Median

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

# Statistics for data1
print("Data 1")
print("Mean:", np.mean(data1))
print("Median:", np.median(data1))
print("Mode:", statistics.mode(data1))

print()

# Statistics for data2
print("Data 2")
print("Mean:", np.mean(data2))
print("Median:", np.median(data2))
print("Mode:", statistics.mode(data2))

# The mean for data2 is much larger than the median because the value
# 150 is an outlier that pulls the mean upward. The median is less
# affected by extreme values, so it better represents the center of
# this skewed dataset.            

# ============================================================
#                  Hypothesis Testing Review
# ============================================================

# ------------- Q1 -  independent samples t-test ----------------
from scipy import stats
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# Perform independent samples t-test
t_statistic, p_value = stats.ttest_ind(group_a, group_b)

print("t-statistic:", t_statistic)
print("p-value:", p_value)

# --------------- Q2 - Is the p value statistically sinificant -------------

alpha = 0.05

# p_value = 1.5471178249432405e-06

if p_value < alpha:
    print("The result is statistically significant.")
else:
    print("The result is not statistically significant.")

# ------------------ Q3 - Paired t-test ----------------------

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

# Perform paired t-test
t_statistic, p_value = stats.ttest_rel(before, after)

print("t-statistic:", t_statistic)
print("p-value:", p_value)

# Results prited 
# t-statistic: -15.1744244666721
# p-value: 1.298717471864237e-06

# The negative t-statistic indicates that the before scores are, on average, lower than the after scores
# Since the p-value is much smaller than 0.05, the difference is statistically significant

# ------------------- Q4 - One-sample t-test ---------------------------

scores = [72, 68, 75, 70, 69, 74, 71, 73]

# One-sample t-test against population mean = 70
t_statistic, p_value = stats.ttest_1samp(scores, 70)

print("t-statistic:", t_statistic)
print("p-value:", p_value)

# t-statistic: 1.7320508075688774
# p-value: 0.12687036692367085

# Since the p-value (~0.17) is greater than 0.05, the result is not statistically significant
# So we do not have enough evidence to say the mean differs from 70.

# --------------------- Q5 - One-tailed independent t-test --------------------
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# One-tailed independent t-test (group_a < group_b)
t_statistic, p_value = stats.ttest_ind(group_a, group_b, alternative='less')

print("p-value:", p_value)

# 7.735589124716202e-07
# This p-value is extremely small. So we reject the null hypothesis.
# There is strong evidence that group_a scores are significantly less than group_b scores

# ----------------------- Q6 - Conclusion ----------------------------

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_statistic, p_value = stats.ttest_ind(group_a, group_b)

alpha = 0.05

if p_value < alpha:
    print("Group A scores are significantly lower than Group B scores, and this difference is unlikely to be due to chance.")
else:
    print("There is no clear evidence that Group A scores are different from Group B scores; the observed difference may be due to chance.")

# =========================================================
#                     Correlation Review
# =========================================================

# ------------------ Q1 - Pearson correlation using numpy --------------------

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Expected correlation: 1.0 because y is exactly 2 times x

correlation_matrix = np.corrcoef(x, y)

print("Correlation matrix:")
print(correlation_matrix)

print("Correlation coefficient:", correlation_matrix[0, 1])

# ------------------ Q2 - Use pearsonr() from scipy.stats -----------------------

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]

# Compute Pearson correlation
correlation, p_value = stats.pearsonr(x, y)

print("Correlation coefficient:", correlation)
print("p-value:", p_value)

# Correlation coefficient: -0.9757575757575758
# p-value: 1.4675461874041921e-06
# The correlation coefficient is close to -1, indicating a very strong negative linear relationship
# The very small p-value indicates that this correlation is statistically significant and is unlikely to have occurred by chance.

# ------------------ Q3 - use df.corr() to compute the correlation matrix -----------------

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 72, 80],
    "age": [25, 30, 22, 35, 28]
}

df = pd.DataFrame(people)

# Compute the correlation matrix
correlation_matrix = df.corr()

print(correlation_matrix)

# -------------- Q4- Create a scatter plot ----------------------

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

# Create scatter plot
plt.scatter(x, y)

plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")

plt.show()


# ----------------- Q5 - create a heatmap ------------------

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 72, 80],
    "age": [25, 30, 22, 35, 28]
}

df = pd.DataFrame(people)

# Compute the correlation matrix
correlation_matrix = df.corr()

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True)

plt.title("Correlation Heatmap")
plt.show()

# =====================================================
#                      Pipelines
# ======================================================

#  --------------- Q1 - Create a pipeline --------------------

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# Create a pandas Series
def create_series(arr):
    return pd.Series(arr, name="values")

# Remove missing values
def clean_data(series):
    return series.dropna()

# Compute summary statistics
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

# Run the complete pipeline
def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary

# Call the pipeline and print the results
result = data_pipeline(arr)

for key, value in result.items():
    print(f"{key}: {value}")

