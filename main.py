import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# STEP 1: LOAD THE DATASET

# We are loading the Telco Customer Churn dataset using pandas.
# This dataset contains information about customers such as:
# - their contract type
# - monthly charges
# - how long they've stayed (tenure)
# - and whether they left the company (Churn)

# We store it in a variable called "df" which stands for "DataFrame" (basically like a table).
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# STEP 2: BASIC DATA CLEANING

# Some values in TotalCharges are stored as text instead of numbers.
# This happens because the dataset contains missing or irregular entries.

# Machine learning models and numerical analysis require all values that represent money or quantities to be numeric.
# So we convert TotalCharges into numbers.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# After conversion, some rows become empty (NaN).
# We remove these rows because incomplete data can break analysis or produce incorrect results in machine learning models.
df = df.dropna()

# Remove customer ID because it is only an identifier and does not help predict churn
df = df.drop("customerID", axis=1)


# STEP 3: QUICK DATA OVERVIEW (UNDERSTANDING THE DATA)

# Show first 5 rows to understand what the dataset looks like.
# This helps us verify that the data loaded correctly.
print(df.head())

# Show number of rows and columns:
# - rows = number of customers
# - columns = number of features (information about each customer)
print(df.shape)

# Show all column names to understand available features.
# This helps us decide which variables are useful for analysis.
print(df.columns)


# STEP 4: UNDERSTANDING THE TARGET VARIABLE (CHURN)
# "Churn" is the most important column in this dataset.
# It tells us whether a customer:
# - "Yes" → left the company
# - "No" → stayed with the company

# Count how many customers stayed vs left
print(df["Churn"].value_counts())

# Calculate churn rate (percentage of customers who left)
# This gives a high-level business metric:
# Example: if churn rate is 26%, it means 26 out of 100 customers left
churn_rate = (df["Churn"] == "Yes").mean() * 100
print(churn_rate)


# STEP 5: VISUALIZING OVERALL CHURN DISTRIBUTION
# This graph shows the total number of customers who stayed vs left.
# It helps us quickly understand whether churn is a big problem or not.
sns.countplot(x="Churn", data=df)
plt.show()


# STEP 6: CONTRACT TYPE VS CHURN (KEY BUSINESS INSIGHT)

# This graph compares contract types (Month-to-month, 1-year, 2-year) against churn behavior.
# This is to answer: "Do customers with short contracts leave more often?"
plt.figure(figsize=(10,5))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract Type vs Churn")
plt.show()



# STEP 7: MONTHLY CHARGES VS CHURN
# This boxplot helps us understand whether pricing affects churn.

# It compares:
# - Monthly charges of customers who stayed
# - Monthly charges of customers who left

# If churned customers pay more, pricing might be a factor.
plt.figure(figsize=(10,5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# Summary statistics for Monthly Charges by churn group
# This gives exact numerical values (mean, median, quartiles)
print(df.groupby("Churn")["MonthlyCharges"].describe())


# STEP 8: TENURE VS CHURN (CUSTOMER LOYALTY ANALYSIS)

# This helps answer: "Are long-term customers more loyal?"
plt.figure(figsize=(10,5))
sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure vs Churn")
plt.show()

# Summary statistics for tenure by churn group
print(df.groupby("Churn")["tenure"].describe())


# STEP 9: DATA TYPE CHECK (IMPORTANT BEFORE MACHINE LEARNING)
# This shows the type of each column:
# - int/float = numerical data (good for ML)
# - object (str) = categorical/text data (needs encoding)

# Machine learning models cannot directly understand text, so we check this before building predictive models.
print(df.dtypes)


# STEP 10: CONVERT TARGET VARIABLE (CHURN) INTO NUMBERS

# Machine learning models cannot understand words like "Yes" or "No".
# So we assign:
# - No  → 0 (customer stayed)
# - Yes → 1 (customer left)

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

print(df["Churn"].head())

print(df.select_dtypes(include="str").columns)


# STEP 11: ENCODE ALL CATEGORICAL FEATURES

# Machine learning models cannot understand text columns.
# So we convert all categorical variables into numeric form using one-hot encoding.

df = pd.get_dummies(df, drop_first=True)

# Show final dataset after encoding
print(df.head())

# Show final number of features
print(df.shape)

df = df.astype(int)

# STEP 12: TRAIN-TEST SPLIT (IMPORTANT FOR MODEL EVALUATION)
# Before building machine learning models, we split the dataset into:
# - Training set (80% of data) → used to train the model
# - Test set (20% of data) → used to evaluate the model's performance on unseen data

from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data into training and testing sets
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 13: TRAIN THE FIRST AI MODEL
# We will use Logistic Regression, which is a common algorithm for binary classification problems like churn prediction.
# Logistic Regression will learn patterns in the training data to predict whether a customer will churn (1) or not (0).
# We set max_iter=1000 to ensure the model has enough iterations to converge during training.

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# STEP 14: MAKE PREDICTIONS WITH THE MODEL
# After training the model, we use it to predict churn on the test set.
# The model will output predicted probabilities of churn, which we can convert to binary predictions (0 or 1) based on a threshold (usually 0.5).
y_pred = model.predict(X_test)

# STEP 15: EVALUATE MODEL PERFORMANCE
# We will evaluate the model using accuracy, which measures the percentage of correct predictions.
from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))