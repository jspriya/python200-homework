import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
  
# fetch dataset 
spambase = fetch_ucirepo(id=94) 
  
# data (as pandas dataframes) 
X = spambase.data.features 
y = spambase.data.targets 

df = pd.concat([X, y], axis=1)

#print(df.shape)
#print(df.head())
#print(df.info()) 
# metadata 
#print(spambase.metadata) 
  
# variable information 
#print(spambase.variables) 

#print(X.head())
#print(y.head())

#print("X shape:", X.shape)
#print("y shape:", y.shape)

# --------- Task1: Load and Explore ---------

# -------------------------
# Class balance
# -------------------------

print("\nSpam label counts:")
print(df["Class"].value_counts())

print("\nSpam label percentages:")
print(df["Class"].value_counts(normalize=True) * 100)

# The dataset contains both spam and ham emails, but the classes are not
# perfectly balanced. Because of this, accuracy alone may not tell the full
# story. A model could achieve a high accuracy by mostly predicting the
# majority class, so metrics like precision, recall, and F1-score are also
# important.

# -------------------------
# Feature distributions
# -------------------------

features = [
    "word_freq_free",
    "char_freq_!",
    "capital_run_length_total"
]

for feature in features:

    plt.figure(figsize=(6, 4))

    df.boxplot(
        column=feature,
        by="Class"
    )

    plt.title(f"{feature} by Spam vs Ham")
    plt.suptitle("")   # remove pandas default title
    plt.xlabel("Spam Label (0 = Ham, 1 = Spam)")
    plt.ylabel(feature)

    plt.savefig(f"outputs/{feature}_boxplot.png")

    plt.show()

# The boxplots show that spam emails generally have higher values for
# word_freq_free and char_freq_! compared to ham emails. Spam messages often
# use promotional words and punctuation patterns. However, there is overlap
# between the classes, meaning no single feature perfectly separates spam
# from ham.

# For capital_run_length_total:
# Spam emails tend to have longer sequences of capital letters, but the
# distributions still overlap. This suggests that the feature is useful but
# should be combined with other features for classification.

# Reason for zero-heavy distribution:
# Many word-frequency features have a large number of zero values because most
# emails do not contain a particular word. This creates sparse data, where
# many features are inactive for each email.

# Reason for variation in feature scales:
# Feature scales vary because different features measure different things.
# Word and character frequencies are percentages and usually have small values,
# while features like capital_run_length_total are counts and can reach much
# larger values. This difference in scale matters for models that depend on
# distances or optimization, such as KNN and Logistic Regression, which usually
# benefit from feature scaling. Tree-based models are less affected because
# they split data using thresholds rather than distances.

# ---------- Task2 : Prepare the Data

# Separate features and target

X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split [use the standard 80/20 split]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)

# stratify=y keeps the same spam/ham proportion in both training and test sets.
# This is important because the dataset is somewhat imbalanced.

# Scale the data
scaler = StandardScaler()

# Fit only on X_train
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# The scaler is fitted only on the training data to prevent information
# from the test set leaking into the model. The test set should represent
# unseen data.

# PCA
pca = PCA()

# Fit only on training scaled data:

pca.fit(X_train_scaled)

# Calculate cumulative explained variance
pca.explained_variance_ratio_

# Create cumulative values:
cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

# Find the number of components needed for 90% variance:

n = np.argmax(cumulative_variance >= 0.90) + 1

print("Number of components needed for 90% variance:", n)

# Plot cumulative explained variance
plt.figure(figsize=(8,5))

plt.plot(
    range(1, len(cumulative_variance)+1),
    cumulative_variance
)

plt.axhline(
    y=0.90,
    linestyle="--"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")

plt.savefig(
    "outputs/pca_explained_variance.png"
)

plt.show()

# Transform data using PCA - Create reduced data
X_train_pca = pca.transform(X_train_scaled)[:, :n]

X_test_pca = pca.transform(X_test_scaled)[:, :n]

# Task2 Summary:
# Decision tree models will use the original unscaled features because trees
# split data based on thresholds and are not sensitive to feature magnitude.
# KNN and logistic regression will use scaled data because both are affected
# by feature magnitude. PCA is applied only after scaling because PCA is based
# on variance, and large-scale features would dominate the principal components.
# Both scaling and PCA are fitted only on training data to prevent test data
# information leakage.

# ------------  Task3 --------------

# 1. K-Nearest Neighbors (KNN) on unscaled data 
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("K-Nearest Neighbors (KNN) on unscaled data")
print("--------------------------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#  2. KNN on scaled data

knn_scaled = KNeighborsClassifier(n_neighbors=5)

knn_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("KNN on scaled data")
print("-------------------")
print("Accuracy:", accuracy_score(y_test, y_pred_scaled))
print(classification_report(y_test, y_pred_scaled))

# 3. KNN on PCA data

# Use the PCA transformed data

knn_pca = KNeighborsClassifier(n_neighbors=5)

knn_pca.fit(X_train_pca, y_train)

y_pred_pca = knn_pca.predict(X_test_pca)


print("KNN on PCA data")
print("-------------------")
print("Accuracy:", accuracy_score(y_test, y_pred_pca))
print(classification_report(y_test, y_pred_pca))

# 4. Decision Tree

depths = [3, 5, 10, None]

print("Decision Tree")
print("--------------")
for depth in depths:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test)

     
    print(f"Depth: {depth}")
    print("Train accuracy:", train_acc)
    print("Test accuracy:", test_acc)

# Use the chosen depth for the final model
tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

print("\nDecision Tree (Final Model)")
print("----------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# As tree depth increases, training accuracy continues to improve,
# but test accuracy eventually levels off or decreases. This indicates
# overfitting. A depth of 5 provides a good balance between model
# complexity and generalization, so it would be the preferred choice
# for deployment.

# Print the top 10 feature important features

feature_names = X.columns

tree_importance = (
    pd.Series(tree.feature_importances_, index=feature_names)
    .sort_values(ascending=False)
)

print("\nTop 10 Decision Tree Features")
print(tree_importance.head(10))


# 5. Random Forest

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Random Forest")
print("-------------")
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Print the top 10 important features

rf_importance = (
    pd.Series(rf.feature_importances_, index=feature_names)
    .sort_values(ascending=False)
)

print("\nTop 10 Random Forest Features")
print(rf_importance.head(10))

# Bar chart

plt.figure(figsize=(10,6))

rf_importance.head(10).plot(kind="bar")

plt.title("Top 10 Random Forest Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()

plt.savefig("outputs/feature_importances.png")

plt.show()

# Both models identify char_freq_$, char_freq_!, word_freq_remove,
# word_freq_free, and capital letter features as important predictors.
# The Decision Tree relies much more heavily on char_freq_$, while the
# Random Forest spreads importance across several features because it
# averages many trees.
#
# These results match my intuition because spam emails often contain
# promotional words, excessive punctuation, and unusual capitalization.

# The Decision Tree and Random Forest agree on several important features,
# including char_freq_$, word_freq_remove, char_freq_!, word_freq_free,
# and capital_run_length_total. However, their rankings are different.
# The Decision Tree relies heavily on char_freq_$ as its primary split,
# while the Random Forest distributes importance more evenly across multiple
# features because it combines predictions from many trees.
#
# These results match my intuition about spam emails. Spam messages often
# contain promotional words such as "free" and "remove", excessive punctuation
# such as "$" and "!", and unusual capitalization patterns. Features related
# to money symbols, marketing language, and capital letter usage are therefore
# strong indicators of spam.


#Agreement between models:
#---------------------------

#Both models found:

#Feature	Decision Tree	Random Forest
#char_freq_$	        #1	    #2
#word_freq_remove	    #2	    #3
#char_freq_!	        #3	    #1
#word_freq_free	        #5	    #4
#Capital letter
#features	        Important	Important

#So they clearly agree on the general spam signals.

#Difference between Decision Tree and Random Forest:
#---------------------------------------------------

#The Decision Tree has:

#char_freq_$ = 0.461

#which is extremely high. That means one tree found a split involving $ that was very powerful for this 
#particular training set.

#The Random Forest has:

#char_freq_$ = 0.103
#char_freq_! = 0.114

#because it averages across many trees and many random feature subsets. It is less likely to over-rely 
#on one feature.

#The Decision Tree appears to rely heavily on a single feature (char_freq_$),
#which may make it more sensitive to changes in the training data. The
#Random Forest provides a more balanced view of feature importance because
#it averages many different trees.


# 6. Logistic Regression (scaled)

lr = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)

print("Logistic Regression (scaled)")
print("-----------------------------")
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Logistic Regression (PCA)

lr.fit(X_train_pca, y_train)

y_pred = lr.predict(X_test_pca)

print("Logistic Regression (PCA)")
print("-----------------------------")
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Based on the model results, the Random Forest is clearly the best-performing model.

#Here's a comparison:

#Model	            Accuracy	   Notes
#-----------------------------------------
#KNN (unscaled)	    0.799	       Poor performance because KNN is sensitive to feature scales.
#KNN (scaled)	    0.908	       Scaling greatly improved performance.
#KNN (PCA)	        0.907	       Almost identical to scaled KNN; PCA did not provide additional benefit.
#Decision Tree	    0.911	       Highest accuracy at max_depth=None, but the perfect training accuracy (0.9997) indicates overfitting.
#Random Forest	    0.946	       Best overall accuracy and balanced precision/recall.
#Logistic 	    
#Regression (scaled) 0.929	       Strong performance.
#Logistic 	    
#Regression(PCA)     0.919	       Slightly worse than without PCA.

#Scaling greatly improved KNN performance because KNN relies on distance
#calculations. Logistic Regression also performed slightly better on the
#scaled data than on the PCA-reduced data, suggesting that PCA removed a
#small amount of useful information. This matches the expectation from
#Task 2 that PCA may help simplify the data but does not always improve
#classification performance.

#For a spam filter, accuracy alone is not the most important metric.
#I would prioritize minimizing false positives, since incorrectly marking
#a legitimate email as spam could cause users to miss important messages.
#Although false negatives allow spam into the inbox, users can usually
#delete unwanted emails more easily than recovering legitimate emails
#that were filtered out.

# 8. Confusion Matrix
ConfusionMatrixDisplay.from_estimator(
    rf,
    X_test,
    y_test,
    display_labels=["Ham", "Spam"],
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")

plt.savefig("outputs/best_model_confusion_matrix.png")

plt.show()

#True Negatives (540): 540 legitimate emails (Ham) were correctly identified as Ham.

#True Positives (331): 331 spam emails were correctly identified as Spam.

#False Positives (18): 18 legitimate emails were misclassified as Spam (false alarms).

#False Negatives (32): 32 spam emails were misclassified as Ham (missed spam).

#Key Takeaways
#---------------
#1. High Overall Accuracy: Out of 921 total test samples ($540 + 18 + 32 + 331$), the model got 871 correct, 
#which is roughly 94.6% accuracy.

#2. Low False Positive Rate: Only 18 out of 558 legitimate emails ($~3.2\%$) were incorrectly flagged as spam.
#This is important because blocking regular emails is usually worse than missing a spam email.

#3. Slightly More False Negatives: The model missed 32 spam emails ($~8.8\%$ of total spam), 
#sending them to the primary inbox instead of the spam folder.

# -------------------------
# Task 4: Cross Validation
# -------------------------

print("\nTask 4: Cross Validation Results")
print("--------------------------------")


# 1. KNN unscaled

knn_cv = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(
    knn_cv,
    X_train,
    y_train,
    cv=5
)

print("\nKNN (unscaled)")
print("Fold scores:", scores)
print("Mean accuracy:", scores.mean())
print("Std deviation:", scores.std())

# 2. KNN scaled

knn_scaled_cv = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(
    knn_scaled_cv,
    X_train_scaled,
    y_train,
    cv=5
)

print("\nKNN (scaled)")
print("Fold scores:", scores)
print("Mean accuracy:", scores.mean())
print("Std deviation:", scores.std())

# 3. KNN PCA

knn_pca_cv = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(
    knn_pca_cv,
    X_train_pca,
    y_train,
    cv=5
)

print("\nKNN (PCA)")
print("Fold scores:", scores)
print("Mean accuracy:", scores.mean())
print("Std deviation:", scores.std())

# 4. Decision Tree

tree_cv = DecisionTreeClassifier(
    max_depth=5,      # replace with your chosen depth
    random_state=42
)

scores = cross_val_score(
    tree_cv,
    X_train,
    y_train,
    cv=5
)

print("\nDecision Tree")
print("Fold scores:", scores)
print("Mean accuracy:", scores.mean())
print("Std deviation:", scores.std())

# 5. Random Forest

rf_cv = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

scores = cross_val_score(
    rf_cv,
    X_train,
    y_train,
    cv=5
)

print("\nRandom Forest")
print("Fold scores:", scores)
print("Mean accuracy:", scores.mean())
print("Std deviation:", scores.std())

# 6. Logistic Regression scaled

lr_cv = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

scores = cross_val_score(
    lr_cv,
    X_train_scaled,
    y_train,
    cv=5
)

print("\nLogistic Regression (scaled)")
print("Fold scores:", scores)
print("Mean accuracy:", scores.mean())
print("Std deviation:", scores.std())

# 7. Logistic Regression PCA

scores = cross_val_score(
    lr_cv,
    X_train_pca,
    y_train,
    cv=5
)

print("\nLogistic Regression (PCA)")
print("Fold scores:", scores)
print("Mean accuracy:", scores.mean())
print("Std deviation:", scores.std())



# Cross-validation results confirm the findings from the single train/test split.
# Random Forest achieved the highest mean accuracy (0.954), making it the best
# performing classifier overall. Logistic Regression with scaled data was the
# second strongest model with a mean accuracy of 0.924.
#
# Logistic Regression with PCA had the lowest standard deviation (0.0034),
# meaning it was the most stable model across the five folds. However, its
# accuracy was lower than Random Forest. Random Forest also showed relatively
# low variance (std = 0.0133), which indicates that it generalizes well across
# different training splits.
#
# The model ranking is mostly consistent with the single train/test split:
# Random Forest remains the strongest model, followed by Logistic Regression,
# while KNN and Decision Tree perform lower. This increases confidence that
# the original results were not caused by a lucky train/test split.
#
# PCA slightly improved KNN performance compared to scaled data (0.908 vs
# 0.905), which matches the hypothesis that dimensionality reduction can help
# distance-based models. However, PCA reduced Logistic Regression performance
# compared to using all scaled features (0.915 vs 0.924), suggesting that some
# useful information was removed when reducing dimensions.
#
# For a spam filter, accuracy alone is not the only important metric. I would
# pay close attention to false positives (legitimate emails incorrectly marked
# as spam), because losing an important email can be more harmful than allowing
# some spam messages through. A good spam filter should balance precision and
# recall rather than optimizing accuracy alone.


#Actual numbers:
#---------------
#Best accuracy:

# Random Forest: 95.43%
# Logistic Regression scaled: 92.36%
# Logistic Regression PCA: 91.49%
# Decision Tree: 90.73%
# KNN PCA: 90.84%
# KNN scaled: 90.46%
# KNN unscaled: 79.43%

# Most stable model (lowest variance)
# ------------------------------------

#Strictly by standard deviation:

#Model	                Std Dev
#Logistic Regression PCA	0.0034 
#KNN PCA	                0.0094
#KNN scaled	            0.0094
#Logistic Regression scaled	0.0097
#Random Forest	        0.0133
#Decision Tree	        0.0158
#KNN unscaled	        0.0182

#Conclusion:
#-----------
#Most accurate: Random Forest
#Most stable: Logistic Regression with PCA

# ----------------------------------
#      Task 5
#-----------------------------------

# -------------------------
# Random Forest Pipeline
# -------------------------

rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)

rf_pred = rf_pipeline.predict(X_test)

print("\nRandom Forest Pipeline")
print("----------------------")
print(classification_report(y_test, rf_pred))

# -------------------------
# Logistic Regression Pipeline
# -------------------------

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="liblinear"
    ))
])

lr_pipeline.fit(X_train, y_train)

lr_pred = lr_pipeline.predict(X_test)

print("\nLogistic Regression Pipeline")
print("----------------------------")
print(classification_report(y_test, lr_pred))

print("Random Forest pipeline accuracy:",
      rf_pipeline.score(X_test, y_test))

print("Logistic Regression pipeline accuracy:",
      lr_pipeline.score(X_test, y_test))

# The two pipelines do not have the same structure because different models
# have different preprocessing requirements. The Random Forest pipeline only
# contains the classifier because tree-based models are not sensitive to
# feature scaling. The Logistic Regression pipeline includes StandardScaler
# because distance and coefficient-based models are affected by feature scale.
#
# Packaging models as pipelines reduces the chance of preprocessing mistakes
# and prevents data leakage because transformations are automatically fitted
# only on the training data. Pipelines also make the model easier to share,
# reproduce, and deploy because the entire prediction process is stored as
# one object.