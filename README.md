# Customer-Churn-Project

# Customer Churn Prediction (Telecom Dataset)

## Overview

This project focuses on predicting customer churn using a machine learning approach. Customer churn refers to customers stopping their subscription or service usage. Understanding churn is important for businesses because retaining existing customers is more cost-effective than acquiring new ones.

Using the Telco Customer Churn dataset, this project analyzes customer behavior patterns and builds a classification model to predict whether a customer will churn or not.

---

## Objective

The main objectives of this project are:

- Analyze customer data and identify patterns related to churn
- Perform data cleaning and preprocessing
- Conduct exploratory data analysis (EDA)
- Build a machine learning model for churn prediction
- Evaluate model performance using classification metrics

---

## Dataset Information

This project uses the **Telco Customer Churn dataset** sourced from Kaggle.

📌 Source: IBM Sample Dataset (available on Kaggle)

The dataset contains customer-level information from a telecom company, including:

- Customer demographics (gender, senior citizen, partner, dependents)
- Account information (tenure, contract type, payment method)
- Services subscribed (internet service, phone service, streaming services)
- Billing details (monthly charges, total charges)
- Target variable: **Churn (Yes/No)**

This dataset is widely used for binary classification and customer retention analysis.

---

## Technologies Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Data Preprocessing

The following preprocessing steps were performed:

- Converted `TotalCharges` from string to numeric format
- Handled missing values by removing incomplete rows
- Dropped irrelevant column (`customerID`)
- Converted categorical variables using one-hot encoding
- Converted target variable:
  - No → 0 (Customer stayed)
  - Yes → 1 (Customer churned)

---

## Exploratory Data Analysis (EDA)

Key insights discovered:

- Customers with **month-to-month contracts** have the highest churn rate
- Customers with **low tenure** are more likely to churn
- Higher **monthly charges** are associated with increased churn probability
- Long-term customers are more likely to stay

### Visualizations used:
- Count plots (churn distribution, contract type)
- Box plots (monthly charges vs churn, tenure vs churn)

---

## Machine Learning Model

### Model Used:
- Logistic Regression

### Why Logistic Regression?
- Suitable for binary classification problems
- Simple and interpretable baseline model
- Efficient for small-to-medium datasets

### Training Process:
- Split dataset into training and testing sets
- Applied one-hot encoding for categorical features
- Trained Logistic Regression model on processed data
- Evaluated using test data predictions

---

## Model Performance

- **Accuracy:** ~78.8%
- **Evaluation Method:** Confusion Matrix

### Confusion Matrix:
[[True Negatives False Positives]
[False Negatives True Positives]]


This shows how well the model predicts both churn and non-churn customers.

---

## 📌 Key Results

- Contract type is a strong indicator of churn behavior
- Customers with shorter tenure are more likely to churn
- Higher monthly charges increase churn probability
- The model provides a good baseline for predicting churn

---

## Business Impact

This model can help businesses such as telecom companies:

- Identify customers at risk of leaving
- Improve customer retention strategies
- Optimize marketing campaigns
- Reduce revenue loss caused by churn

---

## Project Structure
│
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── main.py
├── README.md
└── requirements.txt

---

## 📌 Acknowledgements

Dataset originally provided by IBM and accessed via Kaggle for educational and learning purposes.

---

## Author

This project was built as a learning exercise in data science and machine learning, focusing on:

- Data analysis
- Feature engineering
- Classification modeling
- Business insight extraction
